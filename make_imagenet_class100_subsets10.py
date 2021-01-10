import shutil, errno
import os

'''
20201206. This function divides imagenet class100 into multiple subsets. 
The number of images in only train category will be divided into subsets. 
Therefore, effective number of images will be reduced while maintaining #category. 

For exmaple, if category dog has 10 images, and I will divide it into 10 subsets, 
subset 1: [image1] (smallest)
subset 2: [image1, image2] 
...
subset 10: [image1, image2, ..., image10] (largest)
And these subsets will be inclusive and will be saved into different directory. 
'''


def copyanything(src, dst):
    '''2020.11.14
    https://stackoverflow.com/a/1994840
    '''
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise

def copyanything_list(src, dst, fn_list):
    '''2020.12.07
    https://www.tutorialspoint.com/How-to-copy-files-from-one-folder-to-another-using-Python#:~:text=For%20copying%20multiple%20files%20at,source%20and%20destination%20are%20strings.)
    copy files given the fn in the list
    '''
    for fn in fn_list:
        shutil.copy(os.path.join(src, fn), os.path.join(dst, fn))

def get_file_names(src):
    '''2020.12.07
    Get list of file names for a given src
    https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    '''
    onlyfiles = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    return onlyfiles


#source_dir = '/home/libiadm/datasets/ImageNet2012/'
source_dir = '/home/choi574/datasets/ImageNet2012_class100/'
dest_dir = '/home/choi574/datasets/ImageNet2012_class100_ss/'
div_total = 10 # divide by 10
div_subs = [1, 3, 5] # 1/10, 3/10, 5/10 will be made

classes = ["n01644900", "n02096051", "n04366367", "n03544143", "n02105412", "n01914609", "n02105162", "n02132136", "n03026506", "n03063599", "n02815834", "n07802026", "n01968897", "n03788365", "n04443257", "n12998815", "n02454379", "n03991062", "n04332243", "n04254680", "n02097298", "n07590611", "n03680355", "n02165105", "n01491361", "n04120489", "n03742115", "n07880968", "n02808304", "n03888257", "n03095699", "n01494475", "n03673027", "n02488702", "n01871265", "n02104365", "n02281787", "n04118538", "n01828970", "n02837789", "n03127747", "n04005630", "n02115913", "n01514859", "n03452741", "n02107908", "n01847000", "n04200800", "n04153751", "n04389033", "n02487347", "n02769748", "n01843383", "n02219486", "n02009912", "n03676483", "n02797295", "n04417672", "n04591157", "n04229816", "n02058221", "n03814906", "n02097130", "n02939185", "n03710637", "n02116738", "n04418357", "n03775071", "n04328186", "n02090721", "n02667093", "n03929855", "n02089078", "n02389026", "n03388183", "n07613480", "n02749479", "n02174001", "n07932039", "n02112018", "n02398521", "n04069434", "n03838899", "n02233338", "n03207743", "n02791270", "n02114855", "n04204238", "n02342885", "n02110063", "n01518878", "n02099712", "n01704323", "n02168699", "n04238763", "n03494278", "n03980874", "n02097209", "n01616318", "n03131574"]

#### for train ####
for sub in div_subs:
    for n in classes:
        source_inst = os.path.join(source_dir, 'train', n)
        dest_inst = os.path.join(dest_dir, str(sub), 'train', n)
        if not os.path.exists(dest_inst):
            os.makedirs(dest_inst)
        fn = get_file_names(source_inst)
        fn_sub = fn[:int(len(fn)/div_total*sub)]

        print(sub, n, len(fn), len(fn_sub))
        print(source_inst, dest_inst)
        copyanything_list(source_inst, dest_inst, fn_sub)

    #copyanything(source_inst, dest_inst)
print('end train')

