import gzip
from os.path import join, exists
from os import listdir, makedirs
import json
from multiprocessing import Pool
import re
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='folder of data.')
    parser.add_argument('--out_dir', type=str, help='folder of output.')
    args = parser.parse_args()
    return args


replace_xml = {'&lt;': '<', '&gt;': '>', '&quot;': '"',
               '&apos;': '\'', '&amp;': '&'}

'''
找witness的smith-waterman算法似乎提前完成了，这里的数据是已经处理好的数据
只是组织了一下将每个输入文件写入对应的'.x','.y','.z','.x.info','.y.info','.z.info'
分别代表OCR输出，witness和手写部分
'''

def process_file(paras):
    fn, out_fn = paras
    with gzip.open(fn, 'r') as f_:
        content = f_.readlines()
    out_x = open(out_fn + '.x.txt', 'w')  # output file for OCR'd text
    out_y = open(out_fn + '.y.txt', 'w')  # output file for duplicated texts (witnesses)
    out_z = open(out_fn + '.z.txt', 'w')  # output file for manual transcription
    out_z = open(out_fn + '.z.txt', 'w')  # output file for manual transcription
    # output file for the information of OCR'd text, each line contains:
    # (group no., line no., file_id, begin index in file, end index in file,
    # number of witnesses, number of manual transcriptions)
    out_x_info = open(out_fn + '.x.info.txt', 'w')
    # output file for the information of each witness, each line contains:
    # (line no, file_id, begin index in file)
    out_y_info = open(out_fn + '.y.info.txt', 'w')
    # output file for the information of each manual transcription,
    # each line contains: (line no, file_id, begin index in file)
    out_z_info = open(out_fn + '.z.info.txt', 'w')
    cur_line_no = 0
    cur_group = 0
    for line in content:
        line = str(line, encoding='utf-8')
        line = json.loads(line.strip('\r\n'))
        cur_id = line['id']
        lines = line['lines']
        # lines:包括begin, text, witnesses
        for item in lines:
            begin = item['begin']
            text = item['text']  # get the OCR'd text line
            # 对原文中的转义字符进行转换而后，删除换行符
            for ele in replace_xml:
                text = re.sub(ele, replace_xml[ele], text)
            text = text.replace('\n', ' ')  # remove '\n' and '\t' in the text
            text = text.replace('\t', ' ')
            text = ' '.join([ele for ele in text.split(' ')
                             if len(ele.strip()) > 0])
            if len(text.strip()) == 0:
                continue
            out_x.write(str(text.encode('utf-8')) + '\n')
            wit_info = ''
            wit_str = ''
            man_str = ''
            man_info = ''
            num_manul = 0
            num_wit = 0
            if 'witnesses' in item:
                for wit in item['witnesses']:
                    wit_begin = wit['begin']
                    wit_id = wit['id']
                    wit_text = wit['text']
                    for ele in replace_xml:
                        wit_text = re.sub(ele, replace_xml[ele], wit_text)
                    wit_text = wit_text.replace('\n', ' ')
                    wit_text = wit_text.replace('\t', ' ')
                    wit_text = ' '.join([ele for ele in wit_text.split(' ')
                                         if len(ele.strip()) > 0])
                    if 'manual' in wit_id:  # get the manual transcription
                        num_manul += 1
                        man_info += str(wit_id) + '\t' + str(wit_begin) + '\t'
                        man_str += str(wit_text.encode('utf-8')) + '\t'
                    else:  # get the witnesses
                        num_wit += 1
                        wit_info += str(wit_id) + '\t' + str(wit_begin) + '\t'
                        wit_str += str(wit_text.encode('utf-8')) + '\t'
            if len(man_str.strip()) > 0:
                out_z.write(man_str[:-1] + '\n')
                out_z_info.write(str(cur_line_no) + '\t' + man_info[:-1] + '\n')
            if len(wit_str.strip()) > 0:
                out_y.write(wit_str[:-1] + '\n')
                out_y_info.write(str(cur_line_no) + '\t' + wit_info[:-1] + '\n')
            out_x_info.write(
                str(cur_group) + '\t' + str(cur_line_no) + '\t' + str(cur_id) + '\t' + str(begin) + '\t' + str(
                    len(text) + begin) + '\t' + str(num_wit) + '\t' + str(num_manul) + '\n')
            cur_line_no += 1
        cur_group += 1
    out_x.close()
    out_y.close()
    out_z.close()
    out_x_info.close()
    out_y_info.close()
    out_z_info.close()


def merge_file(data_dir, out_dir):  # merge all the output files and information files
    list_file = [ele for ele in listdir(data_dir) if ele.startswith('part')]
    list_out_file = [join(out_dir, 'pair.' + str(i)) for i in range(len(list_file))]
    out_fn = join(out_dir, 'pair')
    out_x = open(out_fn + '.x.txt', 'w')
    out_y = open(out_fn + '.y.txt', 'w')
    out_z = open(out_fn + '.z.txt', 'w')
    out_z_info = open(out_fn + '.z.info.txt', 'w')
    out_x_info = open(out_fn + '.x.info.txt', 'w')
    out_y_info = open(out_fn + '.y.info.txt', 'w')
    last_num_line = 0
    last_num_group = 0
    total_num_y = 0
    total_num_z = 0
    for fn in list_out_file:
        num_line = 0
        for line in open(fn + '.x.txt'):
            out_x.write(line)
            num_line += 1
        for line in open(fn + '.y.txt'):
            out_y.write(line)
        for line in open(fn + '.z.txt'):
            out_z.write(line)
        dict_x2liney = {}
        dict_x2linez = {}
        for line in open(fn + '.y.info.txt'):
            line = line.split('\t')
            line[0] = str(int(line[0]) + last_num_line)
            dict_x2liney[line[0]] = total_num_y
            total_num_y += 1
            out_y_info.write('\t'.join(line))
        for line in open(fn + '.z.info.txt'):
            line = line.split('\t')
            line[0] = str(int(line[0]) + last_num_line)
            dict_x2linez[line[0]] = total_num_z
            total_num_z += 1
            out_z_info.write('\t'.join(line))
        num_group = 0
        for line in open(fn + '.x.info.txt'):
            line = line.strip('\r\n').split('\t')
            cur_group = int(line[0])
            line[0] = str(int(line[0]) + last_num_group)
            line[1] = str(int(line[1]) + last_num_line)
            if line[1] in dict_x2liney:
                line.append(str(dict_x2liney[line[1]]))
            else:
                line[5] = '0'
            if line[1] in dict_x2linez:
                line.append(str(dict_x2linez[line[1]]))
            else:
                line[6] = '0'
            out_x_info.write('\t'.join(line) + '\n')
            if cur_group > num_group:
                num_group = cur_group
        last_num_group += num_group
        last_num_line += num_line
        for post_fix in ['.x', '.y', '.z']:
            os.remove(fn + post_fix + '.txt')
            os.remove(fn + post_fix + '.info.txt')
    out_x.close()
    out_y.close()
    out_z.close()
    out_x_info.close()
    out_y_info.close()
    out_z_info.close()


'''
读取文件名中有‘part’字样的并且为每一个文件在out_dir中生成对应的'pair.'文件名，
如果不存在out_dir则创建之
新建100个进程的进程池
使用进程池将输入文件和输出文件组成元组执行process_file操作
'''

def process_data(data_dir, out_dir):
    list_file = [ele for ele in listdir(data_dir) if ele.startswith('part')]
    list_out_file = [join(out_dir, 'pair.' + str(i)) for i in range(len(list_file))]
    list_file = [join(data_dir, ele) for ele in list_file]
    if not exists(out_dir):
        makedirs(out_dir)
    pool = Pool(100)
    pool.map(process_file, zip(list_file, list_out_file))
    # map(process_file, zip(list_file, list_out_file))
    # merge_file(data_dir, out_dir)


def main():
    args = get_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    process_data(data_dir, out_dir)


if __name__ == '__main__':
    main()