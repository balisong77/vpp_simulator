import csv
import collections
import math

class Node():
    def __init__(self, name, first_packet_clock, packet_clock):
        self.name = name
        # 第一个包的处理所需时钟
        self.first_packet_clock = first_packet_clock
        # 后续包的处理所需时钟
        self.packet_clock = packet_clock
        self.remaing_packets = 0

class VPPSimulator():
    def __init__(self):
        # 是否打印日志
        self.debug = True
        # 在这里控制当前的批大小
        self.batch_size = 64
        self.nodes = [Node(name="ipsec", first_packet_clock={32:4230, 64:2100}, packet_clock={32:4130, 64:2000}), 
                      Node(name="l3", first_packet_clock={32:550, 64:400}, packet_clock={32:471, 64:391})]
        self.trace_file = open("trace.csv", "r")
        self.reader = csv.reader(self.trace_file)
        # 是否结束标记
        self.done = False
        next(self.trace_file, None)

    def close(self):
        self.trace_file.close()

    def reset(self):
        for node in self.nodes:
            node.reamaing_packets = 0

    def calculate_processtime(self, node):
        # 处理所有包所需的时间
        time_cost = 0
        # 每个包的平均等待时间
        avg_lat = 0
        # 本次step处理的包总数
        total_packet = node.remaing_packets
        # 当前节点处理第一个包所需的clock
        first_packet_clock = node.first_packet_clock[self.batch_size]
        # 当前节点处理后续包所需的clock
        packet_clock = node.packet_clock[self.batch_size]

        # RTC 将所有包处理完
        i = 0
        while node.remaing_packets > 0:
            i += 1
            if node.remaing_packets >= self.batch_size:
                # 计算平均等待时间，当前批次的等待时间 = 前面所有批次包的总处理时间
                avg_lat += time_cost / total_packet
                # 当前批次处理bath_size个包
                node.remaing_packets -= self.batch_size
                # 第一个包的处理时间（指令缓存未命中，耗时稍长）
                time_cost += first_packet_clock
                # 处理后续包的总时间
                time_cost += packet_clock * (self.batch_size - 1)
                if self.debug : print(f"batch: {i}, time cost: {time_cost}, avg latency: {avg_lat}")
            else:
                # 如果当前剩余包不满足一个batch_size，则将剩余包全部处理完
                avg_lat += time_cost / total_packet
                time_cost += first_packet_clock
                time_cost += packet_clock * (node.remaing_packets - 1)
                node.remaing_packets = 0
                if self.debug : print(f"batch: {i}, time cost: {time_cost}, avg latency: {avg_lat}")
        return time_cost, avg_lat

    def step(self):
        # trace文件中，每一行为：当前tick的包的数量，当前不同IPSec流量比例(剩下的是L3)
        line = next(self.reader, None)
        if line is None:
            self.done = True
            return
        packet_nums = line[0]
        ipsec_ratio = line[1]

        ipsec_packet_nums = math.ceil(int(packet_nums) * float(ipsec_ratio))
        l3_packet_nums = int(packet_nums) - ipsec_packet_nums
        self.nodes[0].remaing_packets = ipsec_packet_nums
        self.nodes[1].remaing_packets = l3_packet_nums
        # print(self.nodes[0].first_packet_clock)
        # 处理IPSec包和L3包
        if self.debug : print(f"--IPSec packets: {ipsec_packet_nums}, L3 packets: {l3_packet_nums}--")
        for node in self.nodes:
            time_cost, avg_lat = self.calculate_processtime(node)
            if self.debug : print(f"+ Node: {node.name}, time cost: {time_cost}, avg latency: {avg_lat}")

if __name__ == "__main__":
    vpp = VPPSimulator()
    print(f"Current batchsize: {vpp.batch_size}, Start simulation...")

    for i in range(10):
        if vpp.debug : print(f"Step {i}")
        vpp.step()

    # while not vpp.done:
    #     vpp.step()

    vpp.close()