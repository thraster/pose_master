import io
import os
import argparse
from multiprocessing import Process, Queue
import numpy as np
from PIL import Image
import lmdb
from tqdm import trange
import random
from pathlib import Path

def getImagePath(args,image_name):
    
    if os.path.exists( os.path.join(args.dataset_dir, image_name) ):
        return os.path.join(args.dataset_dir, image_name)
    if hasattr(args, 'dataset_dir_list'):
        for n in args.dataset_dir_list:
            if os.path.exists( os.path.join(n, image_name) ):
                return os.path.join(n, image_name)

    # image_name = image_name.replace('.jpg', '.png')
    # if os.path.exists( os.path.join(args.dataset_dir, image_name) ):
    #     return os.path.join(args.dataset_dir, image_name)
    # if hasattr(args, 'dataset_dir_list'):
    #     for n in args.dataset_dir_list:
    #         if os.path.exists( os.path.join(n, image_name) ):
    #             return os.path.join(n, image_name)

    image_name = image_name.replace('.jpg', '.bmp')    
    if os.path.exists( os.path.join(args.dataset_dir, image_name) ):
        return os.path.join(args.dataset_dir, image_name)
    if hasattr(args, 'dataset_dir_list'):
        for n in args.dataset_dir_list:
            if os.path.exists( os.path.join(n, image_name) ):
                return os.path.join(n, image_name)  

    print(f"not find any {image_name}")

    return None
        
def worker(images,args,q,imgsize):
    print(len(images))      
    for impath in images:
        impath = impath.replace('\\','/')
        impathlist = impath.split()
        if len(impathlist) != 2:
            q.put((impath,"NULL", "label"))
            print(f"wrong image path line:{impath}")
            continue
        image_path = getImagePath(args,impathlist[0])
        if image_path:          
            count = 0
            with open(image_path,'rb') as f:                
                count = 1
                #check image and it's size is valid
                with Image.open(image_path) as img:
                    imgsz = np.array(img).shape
                    # print(f"imgsz:", imgsz.size)
                    # print(f"imgsize size:", imgsize.size)
                    if imgsz == imgsize:
                        # if True:
                        #     data = io.BytesIO() 
                        #     # Setting the points for cropped image 
                        #     width, height = img.size  
                        #     left = 0 ; top = 0; right = width; bottom = height / 2                            
                        #     # Cropped image of above dimension 
                        #     # (It will not change orginal image) 
                        #     im1 = img.crop((left, top, right, bottom))
                        #     im1.save(data, 'JPEG') 
                                                   
                        if img.format != "JPEG":
                            data = io.BytesIO()                            
                            img.save(data, 'JPEG')      
                        else:
                            data=io.BytesIO(f.read())

                        label = impathlist[-1]
                        q.put((impathlist[0],data, label))
                        count = 2
                    else:
                        count = 3
                        q.put((impath,"NULL", "label"))
                        print(f"image shape mismatch {imgsz}")

            if count ==0:
                q.put((impath,"NULL", "label"))
                print("file read error:{}".format(impathlist[0]))
            if count ==1:
                q.put((impath,"NULL", "label"))
                print("image read error:{}".format(impathlist[0]))

        else:
            q.put((impath,"NULL", "label"))
            print("no file:{}".format(impathlist[0]))


def main(PrcsImgNum = -1):
    args = parser.parse_args()    
    args.dataset_dir = args.dataset_dir.replace('\\','/')
    args.imagelist_label = args.imagelist_label.replace('\\','/')
    args.lmdb_file = args.lmdb_file.replace('\\','/')
    
    images = open(args.imagelist_label).readlines()
    num_images = len(images)
    if PrcsImgNum > 0:
        num_images = PrcsImgNum
    b = int(np.ceil(num_images / args.num_workers))

    q = Queue()
    write_fail_list = []
    write_list=[]
    clsSet = set()   
    
    
    first_image = images[0].split()[0].replace('\\','/')
    first_image = getImagePath(args,first_image)
    if first_image:
        imgsz = np.array( Image.open( first_image ) ).shape
    else:
        print('can not find the images , please dataset directory  ')
        return

    jobs = []
    for i in range(args.num_workers):
        if i == args.num_workers-1:
            p = Process(target=worker, args=(images[b*i:num_images],args,q,imgsz))
        else:
            p = Process(target=worker, args=(images[b*i:b*(i+1)],args,q,imgsz))
        p.start()
        jobs.append(p)
        
        
    mean_arr = dis_arr = None
    db_count = 0

    map_size=int(1024*1024*1024*490)# window can not grow the dataset file auto, need set a estimated bigger size
    max_dbs=map_size//(1024*1024*1024*10)+1
    env = lmdb.open(args.lmdb_file, map_size=map_size)#,max_dbs=max_dbs) 

    with env.begin(write=True) as txn:
        for idx in trange(num_images):
            name, data, label = q.get()
            if data == 'NULL':
                write_fail_list.append(name)
                continue
            
            image_key = 'image-{}'.format(db_count)
            label_key = 'label-{}'.format(db_count)
            
            re1 = txn.put(image_key.encode(), data.getvalue())
            re2 = txn.put(label_key.encode(), label.encode())
            
            if re1==False or re2==False:                
                write_fail_list.append(name+ " " +label+"\n")
                print(f'write error for image: {name}')
            else:
                db_count = db_count + 1
                write_list.append(name+ " " +label+"\n")  
                clsSet.add(label)
                
                if args.b_mean_div == True:
                    with Image.open(data) as im:
                        sample = np.asarray(im,dtype=float)
                        if mean_arr is None:
                            mean_arr = np.zeros_like(sample,dtype=float)
                            dis_arr = np.zeros_like(sample,dtype=float)                
                        mean_arr += sample
                        dis_arr += np.power(sample,2)
            
        label_cls = 'label-cls'
        txn.put(label_cls.encode(), '{}'.format(len(clsSet)).encode() )
        
        if args.class_num != len(clsSet):
            print(f"class number input {args.class_num} is not same as the real cls number {len(clsSet)}")
            
    if args.b_mean_div == True:
        mean_arr /= num_images
        dis_arr = np.sqrt(dis_arr/num_images - np.power(mean_arr,2))        
        mean_arr = mean_arr.astype(np.float32)
        dis_arr = dis_arr.astype(np.float32)
        mean_arr.tofile(r"mean.bin")
        dis_arr.tofile(r"div.bin")
        

    for p in jobs:
        p.join(1000)   
    
        
    if len(write_fail_list)>0:
	    with open(os.path.join(args.lmdb_file, "write_fail_list.txt"),"w") as f:
	    	for i in write_fail_list:
	    		f.write(i)  
        
    if len(write_list)>0:
	    with open(os.path.join(args.lmdb_file, "db_list.txt"),"w") as f:
	    	for i in write_list:
	    		f.write(i)
       
    print('Done.')
    print(f"total write {len(write_list)} images, {len(clsSet)} cls")
    if len(write_list) != db_count:
        print(f"strange: write number {len(write_list)} != db_count {db_count}")


 
        

    

    

    
 

                              





if __name__ == "__main__":
    


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str,
                        default=r'E:\U_FaceDB01\webface260m\Images')
    parser.add_argument('--dataset-dir-list', type=list,
                         default=[r'E:\DLTrainSets\c3s112_lank\NewLA_County_retina'                         
                         ])
    parser.add_argument('--imagelist-label', type=str,
                        default=r'E:\U_FaceDB01\webface260m\CLA\webNewLASubNameA2RMNoExist_Caffe_Train_list.txt')
    parser.add_argument('--lmdb-file', type=str, default=r'H:\webNewLASubNameA2RMNoExist_train_lmdb')
    parser.add_argument('--num-workers', type=int, default=16)
    parser.add_argument('--b_mean_div', type=bool, default=False)
    parser.add_argument('--class_num', type=int, default=2412673)  



    main()




   

