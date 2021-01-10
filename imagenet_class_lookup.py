import shutil, errno
import os

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


source_dir = '/mnt/lls/local_export/3/home/choi574/git_libs/misc/map_clsloc.txt'

classes = ["n01644900", "n02096051", "n04366367", "n03544143", "n02105412", "n01914609", "n02105162", "n02132136", "n03026506", "n03063599", "n02815834", "n07802026", "n01968897", "n03788365", "n04443257", "n12998815", "n02454379", "n03991062", "n04332243", "n04254680", "n02097298", "n07590611", "n03680355", "n02165105", "n01491361", "n04120489", "n03742115", "n07880968", "n02808304", "n03888257", "n03095699", "n01494475", "n03673027", "n02488702", "n01871265", "n02104365", "n02281787", "n04118538", "n01828970", "n02837789", "n03127747", "n04005630", "n02115913", "n01514859", "n03452741", "n02107908", "n01847000", "n04200800", "n04153751", "n04389033", "n02487347", "n02769748", "n01843383", "n02219486", "n02009912", "n03676483", "n02797295", "n04417672", "n04591157", "n04229816", "n02058221", "n03814906", "n02097130", "n02939185", "n03710637", "n02116738", "n04418357", "n03775071", "n04328186", "n02090721", "n02667093", "n03929855", "n02089078", "n02389026", "n03388183", "n07613480", "n02749479", "n02174001", "n07932039", "n02112018", "n02398521", "n04069434", "n03838899", "n02233338", "n03207743", "n02791270", "n02114855", "n04204238", "n02342885", "n02110063", "n01518878", "n02099712", "n01704323", "n02168699", "n04238763", "n03494278", "n03980874", "n02097209", "n01616318", "n03131574"]

cls_s = sorted(classes)

with open(source_dir) as f:
    dataset_1000 = f.readlines()


dataset_100 = {}
# n01010101: [dog, 1]
cnt = 0
for d in dataset_1000:
    cn = d.split(' ')[0]
    if cn in classes:
        dataset_100[cn] = [d.split(' ')[2][:-1]]
        cnt = cnt + 1
#print(cnt)

cnt = 0
for c in cls_s:
    dataset_100[c].append(str(cnt))
    cnt = cnt + 1

#print(dataset_100)
#print(cnt)



dataset_100_name = {}
# dog: [n0101010, 1]
for c in cls_s:
    n = dataset_100[c][0]
    dataset_100_name[n] = [c, dataset_100[c][1]]
#print(dataset_100_name)

dataset_100_number = {}
# 1: [n01010101, dog]
for c in cls_s:
    numb = dataset_100[c][1]
    dataset_100_number[numb] = [c, dataset_100[c][0]]
#print(dataset_100_number)

def lookup_number(n):
    return dataset_100_number[str(n)]

def lookup_name(n):
    return dataset_100_name[n]

def lookup_label(n):
    return dataset_100[n]

