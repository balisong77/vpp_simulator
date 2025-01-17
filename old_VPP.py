import collections
import logging
import math
from typing import Any, Dict, SupportsFloat, Tuple
import time

import gym
import numpy as np
from gym import spaces
from gym.core import ActType, ObsType

logger = logging.getLogger(__name__)
handler = logging.FileHandler(f"vpp_old.log", "w")
logger.addHandler(handler)
logger.setLevel(logging.ERROR)


class Node:
    def __init__(self, name, first_packet_clock, packet_clock):
        self.name = name
        # 第一个包的处理所需时钟
        self.first_packet_clock: dict[int, int] = first_packet_clock
        # 后续包的处理所需时钟
        self.packet_clock: dict[int, int] = packet_clock


class Packet:
    def __init__(self, type, enqueue_tick) -> None:
        self.type = type
        self.enqueue_tick = enqueue_tick
        self.end_tick = -1

    def __str__(self) -> str:
        return f"|Packet: type: [{self.type}], enqueue_tick: [{self.enqueue_tick}], end_tick: [{self.end_tick}]|`"


class IncomingPacket:
    def __init__(self, total_count, ipsec_count, l3_count) -> None:
        self.total_count = total_count
        self.ipsec_count = ipsec_count
        self.l3_count = l3_count


class PacketCount:
    def __init__(
        self,
        ipsec_count,
        l3_count,
        ipsec_latency,
        l3_latency,
        incoming_packet: IncomingPacket,
    ) -> None:
        self.ipsec_count = ipsec_count
        self.l3_count = l3_count
        self.ipsec_latency = ipsec_latency
        self.l3_latency = l3_latency
        self.total_count = ipsec_count + l3_count
        self.total_latency = ipsec_latency + l3_latency
        # 当前step内收到的包的数量
        self.incoming_packet = incoming_packet


class Result:
    def __init__(self, total_packet, avg_latency, ipsec_latency, l3_latency) -> None:
        self.total_packet = total_packet
        self.avg_latency = avg_latency
        self.ipsec_latency = ipsec_latency
        self.l3_latency = l3_latency

    def get_reward(self):
        return self.total_packet - self.avg_latency


class VPPSimulatorEnv(gym.Env):
    def __init__(self):
        # 在这里控制当前的批大小
        self.batch_size = 64
        # 当前模拟10000个时钟为一个tick
        self.colck_per_tick = 10000
        # 一个step运行多少个tick
        self.ticks_per_step = 500
        # 2GHz 一秒 2e9个clock
        # 1Mpps 一秒 1e6个包
        # 平均每 10000 clock 会来5个包 (1e6 / 2e9 * 10000)
        self.packet_per_tick = 5
        # ipsec包的占比为40%
        self.ipsec_packet_ratio = 0.4
        # 全局时钟变量
        self.current_tick = 0
        # 当前未处理完的包数量
        self.current_processing_count = 0
        # 记录当前已执行的step数
        self.current_step = 0
        # 攒一批的最大tick等待时间
        self.batch_due = 30
        # 攒包的超时 deadline tick
        self.batch_due_countdown = -1
        # trace模拟流量输入文件，每一行为：当前tick的包的数量，当前不同IPSec流量比例(剩下的是L3)
        self.trace_lines = open("./trace.csv", "r").readlines()[1:]
        self.trace_line_index = 0
        self.result_file = open(f"result_old_{self.batch_size}.csv", "w")
        self.read_from_file = True
        # 每种node的运行时间配置
        self.nodes: Dict[str, Node] = {
            "ipsec": Node(
                name="ipsec",
                first_packet_clock={32: 4130 + 320, 64: 4130 + 320},
                packet_clock={32: 2000 + 50, 64: 2000 + 50},
            ),
            "l3": Node(
                name="l3",
                first_packet_clock={32: 320, 64: 320},
                packet_clock={32: 50, 64: 50},
            ),
        }
        # 是否结束标记
        self.done = False
        # packet缓存队列
        self.packet_queue: collections.deque[Packet] = collections.deque()
        # 状态空间
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([10000, 10000]),
            shape=(2,),
            dtype=np.int32,
        )
        # 动作空间
        self.action_space = spaces.Discrete(2)

    def reset(self):
        # 全局时钟变量
        self.current_tick = 0
        # 当前未处理完的包数量
        self.current_processing_count = 0
        # 记录当前已执行的step数
        self.current_step = 0
        # 攒包的超时 deadline tick
        self.batch_due_countdown = -1
        # 是否结束标记
        self.done = False
        # packet缓存队列
        self.packet_queue: collections.deque[Packet] = collections.deque()
        # self.init_packet_queue()

        # TODO: Uncomment the following line after upgrading to Gymnasium
        # return self._get_obs(), {}
        return self._get_obs()

    def process_batch_packet(self, batch_size):
        logger.debug(
            f"[process_batch_packet] current process batch size: [{batch_size}]"
        )
        # 清除当前攒批的倒计时
        self.batch_due_countdown = -1
        packet_queue = self.packet_queue
        ipsec_count = 0
        l3_count = 0
        # 根据包类型计数
        for i in range(batch_size):
            packet: Packet = packet_queue[i]
            if packet.type == "ipsec":
                ipsec_count += 1
            else:
                l3_count += 1
        # 计算当前批次的处理的总时间
        clock_cost = (
            max(
                0, (ipsec_count - 1) * self.nodes["ipsec"].packet_clock[self.batch_size]
            )
            + max(0, (l3_count - 1) * self.nodes["l3"].packet_clock[self.batch_size])
            + self.nodes["ipsec"].first_packet_clock[self.batch_size]
            + self.nodes["l3"].first_packet_clock[self.batch_size]
        )
        # 计算处理结束的tick
        end_tick = self.current_tick + math.ceil(clock_cost / self.colck_per_tick)
        # 检查计算是否出错
        if end_tick <= self.current_tick:
            logger.error(
                f"[process_batch_packet] end_tick: [{end_tick}] is less than current_tick: [{self.current_tick}], packet_queue: {packet_queue}"
            )
        logger.info(
            f"[process_batch_packet] ipsec: [{ipsec_count}], l3: [{l3_count}], clock cost: [{clock_cost}], end tick: [{end_tick}]"
        )
        # 将这批packet的结束时间设置为end_tick
        for i in range(batch_size):
            packet_queue[i].end_tick = end_tick
            self.current_processing_count = batch_size

    def get_incoming_packet(self) -> IncomingPacket:
        # 从trace文件中获取当前tick的包的数量，和当前的ipsec比例
        if self.read_from_file:
            line = self.trace_lines[self.trace_line_index].strip().split(",")
            self.trace_line_index = (self.trace_line_index + 1) % len(self.trace_lines)
            total_packet_nums: int = int(line[0])
            ipsec_ratio: float = float(line[1])
        # 或通过代码配置获取
        else:
            total_packet_nums = self.packet_per_tick
            ipsec_ratio = self.ipsec_packet_ratio
        ipsec_packet_number: int = round(total_packet_nums * ipsec_ratio)
        l3_packet_number: int = total_packet_nums - ipsec_packet_number
        return IncomingPacket(total_packet_nums, ipsec_packet_number, l3_packet_number)

    def dump_packet_queue(self):
        deque_str = "".join([str(item) for item in self.packet_queue])
        # logger.debug(f"--------------------")
        # logger.debug(f"[dump_packet_queue] tick:[{self.current_tick}]")
        # logger.debug(f"[dump_packet_queue] packet queue: {deque_str}")

    # 第一个tick初始化队列，将packet入队，并计算结束时间
    def init_packet_queue(self):
        incoming_packet: IncomingPacket = self.get_incoming_packet()
        ipsec_packet_number: int = incoming_packet.ipsec_count
        l3_packet_number: int = incoming_packet.l3_count
        packet_queue: collections.deque[Packet] = self.packet_queue
        for i in range(incoming_packet.total_count):
            if i < ipsec_packet_number:
                packet_queue.append(
                    Packet(type="ipsec", enqueue_tick=self.current_tick)
                )
            else:
                packet_queue.append(Packet(type="l3", enqueue_tick=self.current_tick))
        if len(packet_queue) >= self.batch_size:
            self.process_batch_packet(batch_size=self.batch_size)
        else:
            self.process_batch_packet(batch_size=len(packet_queue))
        self.current_tick += 1
        self.dump_packet_queue()

    def run_one_tick(self) -> PacketCount:
        logger.info(
            f"+------------------Current tick: {self.current_tick}------------------+"
        )
        packet_queue: collections.deque[Packet] = self.packet_queue
        # -- 1. 生成当前tick的包 --
        # 将当前tick到的包加入队列右端
        incoming_packet: IncomingPacket = self.get_incoming_packet()
        ipsec_packet_number: int = incoming_packet.ipsec_count
        l3_packet_number: int = incoming_packet.l3_count
        for i in range(incoming_packet.total_count):
            if i < ipsec_packet_number:
                packet_queue.append(
                    Packet(type="ipsec", enqueue_tick=self.current_tick)
                )
            else:
                packet_queue.append(Packet(type="l3", enqueue_tick=self.current_tick))
        logger.debug(
            f"[run_one_tick] push packets ipsec:[{ipsec_packet_number}], l3: [{l3_packet_number}]"
        )
        # 如果当前队列中没有包，直接跳过
        if len(packet_queue) == 0:
            logger.debug(
                f"[run_one_tick] queue length: {len(packet_queue)} nothing to do"
            )
            self.current_tick += 1
            self.dump_packet_queue()
            return PacketCount(0, 0, 0, 0, incoming_packet)
        # -- 2. 处理当前tick的包 --
        # 记录当前tick的包的数量和延迟
        total_packet_current_tick = 0
        total_latency_current_tick = 0
        l3_latency_current_tick = 0
        l3_packet_current_tick = 0
        ipsec_latency_current_tick = 0
        ipsec_packet_current_tick = 0
        # 获取队列左端的第一个packet
        first_packet: Packet = packet_queue[0]
        logger.info(f"[run_one_tick] first packet: {first_packet}")
        # 如果当前正在处理的这批packet的结束时间已到达，将这批packet出队，并计算延迟
        if first_packet.end_tick == self.current_tick:
            for i in range(self.current_processing_count):
                packet: Packet = packet_queue.popleft()
                total_latency_current_tick += packet.end_tick - packet.enqueue_tick
                # 根据包类型分别统计延迟
                if packet.type == "ipsec":
                    ipsec_latency_current_tick += packet.end_tick - packet.enqueue_tick
                    ipsec_packet_current_tick += 1
                else:
                    l3_latency_current_tick += packet.end_tick - packet.enqueue_tick
                    l3_packet_current_tick += 1
            total_packet_current_tick = self.current_processing_count
            avg_latency = total_latency_current_tick / total_packet_current_tick
            logger.info(
                f"[run_one_tick] finish pop packets, avg latency: [{avg_latency}]"
            )
            # 当前批次packet处理完毕，开始进行下一批packet的处理
            if len(packet_queue) >= self.batch_size:
                self.process_batch_packet(batch_size=self.batch_size)
            else:
                # -- 3.如果当前队列中的packet数量不足一个batch，等待下一个tick--
                # 初始化情况，设置batch_due_countdown
                if self.batch_due_countdown == -1:
                    self.batch_due_countdown = self.batch_due
                elif self.batch_due_countdown == 0:
                    self.process_batch_packet(batch_size=len(packet_queue))
                    logger.info(
                        f"[run_one_tick] batch due countdown is 0, process batch packet.."
                    )
                else:
                    logger.info(
                        f"[run_one_tick] queue len: [{len(packet_queue)}] waitting for batch.. batch due countdown: [{self.batch_due_countdown}]"
                    )
                    self.batch_due_countdown -= 1
        # 如果当前队头的packet的结束时间未设置，说明当这批packet需要在当前tick处理
        elif first_packet.end_tick == -1:
            if len(packet_queue) >= self.batch_size:
                self.process_batch_packet(batch_size=self.batch_size)
            else:
                # -- 3.如果当前队列中的packet数量不足一个batch，等待下一个tick--
                # 初始化情况，设置batch_due_countdown
                if self.batch_due_countdown == -1:
                    self.batch_due_countdown = self.batch_due
                elif self.batch_due_countdown == 0:
                    self.process_batch_packet(batch_size=len(packet_queue))
                    logger.info(
                        f"[run_one_tick] batch due countdown is 0, process batch packet.."
                    )
                else:
                    logger.info(
                        f"[run_one_tick] queue len: [{len(packet_queue)}] waitting for batch.. batch due countdown: [{self.batch_due_countdown}]"
                    )
                    self.batch_due_countdown -= 1
        # 如果当前队头的packet的结束时间未到达，继续等待其处理结束
        elif first_packet.end_tick < self.current_tick:
            logger.debug(
                f"[run_one_tick] current tick: [{self.current_tick}] is greater than first packet's end tick: [{first_packet.end_tick}]"
            )
            exit(1)
        # 当前队头的packet的处理结束时间未到达，继续等待
        else:
            logger.debug(
                f"[run_one_tick] queue length: {len(packet_queue)} nothing to do"
            )

        # 本轮处理结束，tick增加
        self.current_tick += 1
        self.dump_packet_queue()
        # 将当前tick处理结束的packet数量，和总延迟返回
        return PacketCount(
            ipsec_packet_current_tick,
            l3_packet_current_tick,
            ipsec_latency_current_tick,
            l3_latency_current_tick,
            incoming_packet,
        )

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        # 修改当前 batchsize
        self.batch_size = action
        # trace文件中，每一行为：当前tick的包的数量，当前不同IPSec流量比例(剩下的是L3)
        packet_count_current_step: PacketCount = PacketCount(
            0, 0, 0, 0, IncomingPacket(0, 0, 0)
        )

        for i in range(self.ticks_per_step):
            pakcket_count_current_tick: PacketCount = self.run_one_tick()
            packet_count_current_step.total_count += (
                pakcket_count_current_tick.total_count
            )
            packet_count_current_step.total_latency += (
                pakcket_count_current_tick.total_latency
            )
            packet_count_current_step.l3_count += pakcket_count_current_tick.l3_count
            packet_count_current_step.l3_latency += (
                pakcket_count_current_tick.l3_latency
            )
            packet_count_current_step.ipsec_count += (
                pakcket_count_current_tick.ipsec_count
            )
            packet_count_current_step.ipsec_latency += (
                pakcket_count_current_tick.ipsec_latency
            )
            packet_count_current_step.incoming_packet.l3_count += (
                pakcket_count_current_tick.incoming_packet.l3_count
            )
            packet_count_current_step.incoming_packet.ipsec_count += (
                pakcket_count_current_tick.incoming_packet.ipsec_count
            )
            packet_count_current_step.incoming_packet.total_count += (
                pakcket_count_current_tick.incoming_packet.total_count
            )

        self.current_step += 1
        avg_latency = (
            0
            if packet_count_current_step.total_count == 0
            else packet_count_current_step.total_latency
            / packet_count_current_step.total_count
        )
        l3_latency = (
            0
            if packet_count_current_step.l3_count == 0
            else packet_count_current_step.l3_latency
            / packet_count_current_step.l3_count
        )
        ipsec_latency = (
            0
            if packet_count_current_step.ipsec_count == 0
            else packet_count_current_step.ipsec_latency
            / packet_count_current_step.ipsec_count
        )
        result: Result = Result(
            total_packet=packet_count_current_step.total_count,
            avg_latency=avg_latency,
            ipsec_latency=ipsec_latency,
            l3_latency=l3_latency,
        )

        # 打印每一步的结果
        self.result_file.write(
            f"{self.current_step},{result.total_packet},{result.avg_latency},{result.ipsec_latency},{result.l3_latency}\n"
        )

        # TODO: Uncomment the following line after upgrading to Gymnasium
        # return self._get_obs(), result.get_reward(), False, False, {}
        return result, result.get_reward(), False, {}

    def render(self):
        pass

    def close(self):
        super().close()

    def _get_obs(self):
        return {"queue_length": len(self.packet_queue)}


# register(
#     id="vpp-simulator",
#     entry_point="envs.VPP:VPPSimulatorEnv",
#     max_episode_steps=300,
# )

if __name__ == "__main__":
    vpp = VPPSimulatorEnv()
    logger.debug(f"Current batchsize: {vpp.batch_size}, Start simulation...")
    # vpp.init_packet_queue()
    vpp.reset()
    # 写csv header
    start_time = time.time()
    vpp.result_file.write("step,total_packet,avg_latency,ipsec_latency,l3_latency\n")
    for i in range(100):
        if i % 300 == 0:
            vpp.reset()
        res = vpp.step(vpp.batch_size)
        result: Result = res[0]
    end_time = time.time()
    print(f"Time cost: {end_time - start_time}")
    # while not vpp.done:
    #     vpp.step()
