import numbers
import os
import queue as Queue
import threading
import lmdb
from PIL import Image
import io
import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


TFS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank, transform=TFS):
        super(MXFaceDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec,
                                                    'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            # print('header0 label', header.label)
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
            # print("Number of Samples:{}".format(len(self.imgidx)))

    def __getitem__(self, index):
        # index =0
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


TFS1 = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
class LMDBDataset(Dataset):
    def __init__(self, root_dir, local_rank, transform=TFS1, target_transform=None,max_worker=1,rgb='BGR',mean_file=None,div_file=None):
        super(LMDBDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.root  = root_dir
        self.local_rank = local_rank
        self.rgb = rgb
        self.PNorm = False
        
        if mean_file and div_file:
            self.mean = np.fromfile(mean_file,dtype=np.float32)
            self.div = np.fromfile(div_file,dtype=np.float32)
            self.PNorm = True

        env = lmdb.open(root_dir, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.n = txn.stat()['entries'] // 2
            self.classes = np.unique([txn.get('label-{}'.format(idx).encode()).decode() for idx in range(self.n)])
        # self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        
        self.txn = None
        
        print('Found {} images belonging to {} classes'.format(self.n, len(self.classes)))

    def _init_db(self):
        self.txn = lmdb.open(self.root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False).begin(write=False)

    def __getitem__(self, index):
        if self.txn is None:
            self._init_db()
        assert index <= self.n, 'index range error'
        data = self.txn.get('image-{}'.format(index).encode())
        label = self.txn.get('label-{}'.format(index).encode()).decode()
        
        sample = Image.open(io.BytesIO(data))       
         
        if self.rgb == 'GRAY': 
            sample = sample.convert('L')
        elif self.rgb == 'BGR': 
            data = np.asarray(sample)
            sample = Image.fromarray(data[:,:,::-1]) 
            
        flip = np.random.choice(2) * 2 - 1
        if flip == 1:
            sample = sample.transpose(Image.FLIP_LEFT_RIGHT)   #PIL image 
            
        if self.PNorm == True:
            data = np.asarray(sample,dtype=np.float32)
            data = (data - self.mean)/self.div
            minv = np.amin(data)
            maxv = np.amax(data)
            data = (255 * (data - minv) / (maxv - minv)).astype(np.uint8)
            sample = Image.fromarray(data)
        
        if self.transform is not None:
            sample = self.transform(sample)    
            
        # target = self.class_to_idx[label]
        target = int(label)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return self.n
    
    def getDBSizeInfo(self):
        return self.n , len(self.classes)


