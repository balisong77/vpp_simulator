import csv
import collections
import math
import logging

class Node():
    def __init__(self, name, first_packet_clock, packet_clock):
        self.name = name
        # 第一个包的处理所需时钟
        self.first_packet_clock : dict[int,int] = first_packet_clock
        # 后续包的处理所需时钟
        self.packet_clock : dict[int,int] = packet_clock

class Packet():
    def __init__(self, type, enqueue_tick) -> None:
        self.type = type
        self.enqueue_tick = enqueue_tick
        self.end_tick = -1

class VPPSimulator():
    def __init__(self):

        # 在这里控制当前的批大小
        self.batch_size = 32
        # 当前模拟10000个时钟为一个tick
        self.colck_per_tick = 10000
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
        self.taken_steps = 0
        # trace模拟流量输入文件
        self.trace_file = open("trace_30.csv", "r")
        self.trace_reader  = csv.reader(self.trace_file)
        self.read_from_file = True
        # 每种node的运行时间配置
        self.nodes : dict[str, Node] = {"ipsec": Node(name="ipsec", first_packet_clock={32:4200+550, 64:2100+400}, packet_clock={32:4100+500, 64:2000+350}), 
                      "l3": Node(name="l3", first_packet_clock={32:550, 64:400}, packet_clock={32:500, 64:350})}
        # 是否结束标记
        self.done = False
        # packet缓存队列
        self.packet_queue : collections.deque[Packet] = collections.deque()
        next(self.trace_reader)

    def __del__(self):
        self.trace_file.close()

    def reset(self):
        pass

    def process_batch_packet(self, batch_size):
        packet_queue = self.packet_queue
        ipsec_count = 0
        l3_count = 0
        # 根据包类型计数
        for i in range(batch_size):
            packet : Packet = packet_queue[i]
            if packet.type == 'ipsec':
                ipsec_count += 1
            else:
                l3_count += 1
        # 计算当前批次的处理的总时间
        clock_cost = max(0, (ipsec_count - 1) * self.nodes["ipsec"].packet_clock[self.batch_size]) \
            + max(0, (l3_count - 1) * self.nodes["l3"].packet_clock[self.batch_size]) \
            + self.nodes["ipsec"].first_packet_clock[self.batch_size] \
            + self.nodes["l3"].first_packet_clock[self.batch_size]
        # 计算处理结束的tick
        end_tick = self.current_tick + math.ceil(clock_cost / self.colck_per_tick)
        logging.debug(f"[process_batch_packet] ipsec: [{ipsec_count}], l3: [{l3_count}], clock cost: [{clock_cost}], end tick: [{end_tick}]")
        # 将这批packet的结束时间设置为end_tick
        for i in range(batch_size):
            packet_queue[i].end_tick = end_tick
            self.current_processing_count = batch_size

    def run_one_tick(self):
        packet_queue : collections.deque[Packet] = self.packet_queue
        # 将当前tick到的包加入队列右端
        if self.read_from_file:
            line = next(self.trace_reader, None)
            if line is None:
                self.done = True
                return
            total_packet_nums : int = int(line[0])
            ipsec_ratio : float = float(line[1])
        else:
            total_packet_nums = self.packet_per_tick
            ipsec_ratio = self.ipsec_packet_ratio
        ipsec_packet_number : int = math.ceil(total_packet_nums * ipsec_ratio)
        l3_packet_number : int = total_packet_nums - ipsec_packet_number
        for i in range(total_packet_nums):
            if i < ipsec_packet_number:
                packet_queue.append(Packet(type = "ipsec",enqueue_tick=self.current_tick))
            else:
                packet_queue.append(Packet(type = "l3",enqueue_tick=self.current_tick))
        logging.debug(f"[run_one_tick] push packets ipsec:[{ipsec_packet_number}], l3: [{l3_packet_number}]")
        # 获取队列左端的第一个packet
        first_packet: Packet = packet_queue[0]
        # 如果当前正在处理的这批packet的结束时间已到达，将这批packet出队，并计算延迟
        if first_packet.end_tick <= self.current_tick:
            avg_latency = 0
            for i in range(self.current_processing_count):
                packet : Packet = packet_queue.popleft()
                avg_latency += (packet.end_tick - packet.enqueue_tick) / self.current_processing_count
            logging.debug(f"[run_one_tick] finish pop packets, avg latency: [{avg_latency}]")
            # 进行下一批packet的处理
            if len(packet_queue) >= self.batch_size:
                self.process_batch_packet(batch_size=self.batch_size)
            else:
                self.process_batch_packet(batch_size=len(packet_queue))
        else:
            logging.debug(f"[run_one_tick] notiong to do")
        # tick增加
        self.current_tick += 1

    def step(self):
        # trace文件中，每一行为：当前tick的包的数量，当前不同IPSec流量比例(剩下的是L3)
        logging.debug(f"+------------------Current tick: {self.current_tick}------------------+")
        self.run_one_tick()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    vpp = VPPSimulator()
    logging.debug(f"Current batchsize: {vpp.batch_size}, Start simulation...")
    for i in range(30):
        vpp.step()

    # while not vpp.done:
    #     vpp.step()
