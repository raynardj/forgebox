import torch
import subprocess

class CudaDevice(object):
    def __init__(self, idx):
        """
        idx: int, cuda device id
        d = CudaDevice(0)
        print(d)
        """
        self.idx = idx
        self.device = torch.device(idx)
        if hasattr(self, "used") == False:
            self.refresh()

    def __repr__(self):
        return f"Device {self.idx}: \n\tname:{self.prop.name}\n\tused:{self.used}MB\tfree:{self.free}MB"

    @property
    def prop(self):
        """
        property on cuda device
        """
        return torch.cuda.get_device_properties(self.idx)

    def __len__(self):
        return int(self.prop.total_memory)

    @property
    def mem(self):
        return f"{self.__len__() // 1024 // 1024} MB"

    def __int__(self):
        return int(self.idx)

    def refresh(self):
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        stat = list(
            int(v) for v in str(gpu_stats).split("\\n")[1 + self.idx].replace(" MiB", "").replace(" ", "").split(","))
        setattr(self, "used", stat[0])
        setattr(self, "free", stat[1])
        return stat

    def __call__(self):
        return self.device


class Cudas(object):
    def __init__(self):
        """
        ch = Cudas()
        dev = ch.idle()()
        some_torch_tensor.to(dev)
        """
        if torch.cuda.is_available() == False:
            return None
        self.counts = torch.cuda.device_count()
        print(f">>> {self.counts} cuda devices found >>>")
        self.devices = list()
        for i in range(self.counts):
            d = CudaDevice(i)
            self.devices.append(d)
            print(f"{d}")
            setattr(self, f"gpu{i}", d)

    def __len__(self):
        """
        counts of the cuda devices numbers
        """
        return self.counts

    def __getitem__(self, idx):
        return self.devices[idx]

    def __repr__(self):
        self.refresh()
        return f"<{self.counts} cuda devices>\n" + "\n".join(list(str(d) for d in (self.devices)))

    def refresh(self):
        """
        refresh the cuda mem stats
        """
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        rows = list(list(int(v) for v in vs.replace(" MiB", "").replace(" ", "").split(",")) for vs in
                    str(gpu_stats).split("\\n")[1:-1])
        for i in range(len(rows)):
            setattr(self.devices[i], "used", rows[i][0])
            setattr(self.devices[i], "free", rows[i][1])
        print("cuda stats refreshed")

    def idle(self):
        """
        find the most idle cuda device
        """
        self.refresh()
        rt = self[0]
        for d in self.devices:
            if d.free > rt.free:
                rt = d
        print(f"Found the most idle GPU: cuda:{rt.idx}, {rt.free} MB Mem remained")
        return rt

CudaHandler = Cudas