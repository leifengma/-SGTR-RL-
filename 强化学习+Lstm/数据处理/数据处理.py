import os
import random
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
# import numpy as np
# import tensorflow as tf

from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
#from sklearn import preprocessing
from tensorflow import keras
#from tqdm import tqdm

pd.set_option ('display.max_columns',None)


def setfig (colume = 1):
	'''
    在绘图前对字体类型、字体大小、分辨率、线宽、输出格式进行设置.
    para colume = 1.半栏图片 7*7cm
                  2.双栏长图 14*7cm
    x轴刻度默认为整数
    手动保存时，默认输出格式为 pdf

    案例 Sample.1:
    setfig(colume=1) # 参数预设置
    plt.plot(x, color='red', linestyle='dashed', label='lengend_label') #此处label指的是Lengend或图例名称
    plt.xlabel('x_axis')
    plt.ylabel('y_axis')
    plt.title('figure_title')
    plt.legend(loc='upper left')
    plt.tight_layout()  #此项必需，以保证绘图正常
    '''
	cm_to_inc = 1 / 2.54  # 厘米和英寸的转换 1inc = 2.54cm
	if colume == 1:
		plt.rcParams['figure.figsize'] = (7 * cm_to_inc,6 * cm_to_inc)  # 单位 inc
	elif colume == 2:
		plt.rcParams['figure.figsize'] = (14 * cm_to_inc,6 * cm_to_inc)
	else:
		pass

	# 对尺寸和 dpi参数进行调整
	plt.rcParams['figure.dpi'] = 300

	# 字体调整
	plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果要显示中文字体,则在此处设为：SimHei
	plt.rcParams['font.weight'] = 'light'
	plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
	plt.rcParams['axes.titlesize'] = 8  # 标题字体大小
	plt.rcParams['axes.labelsize'] = 7  # 坐标轴标签字体大小
	plt.rcParams['xtick.labelsize'] = 7  # x轴刻度字体大小
	plt.rcParams['ytick.labelsize'] = 7  # y轴刻度字体大小
	plt.rcParams['legend.fontsize'] = 6

	# 线条调整
	plt.rcParams['axes.linewidth'] = 1

	# 刻度在内，设置刻度字体大小
	plt.rcParams['xtick.direction'] = 'in'
	plt.rcParams['ytick.direction'] = 'in'

	# 设置输出格式为PDF
	plt.rcParams['savefig.format'] = 'svg'

	# 设置坐标轴范围：如需要可在函数外进行设置
	from matplotlib.ticker import MaxNLocator
	plt.gca ().xaxis.set_major_locator (MaxNLocator (integer=True))  # x轴刻度设置为整数


plt.rcParams['font.sans-serif'] = ['SimHei']  # 显⽰中文
plt.rcParams['axes.unicode_minus'] = False  # 显⽰负号


def get_str_time ():
	s = '_'
	for i in time.localtime (time.time ())[0:6]:
		s += str (i) + '_'
	return s


def make_file(file_dir):
	if not os.path.exists (file_dir):
		os.makedirs (file_dir)


def get_csv (file_path):
	"""
	得到所有csv数据路径
	"""
	root,dir,file = None,None,None
	for root,dir,file in os.walk (file_path):
		root,dir,file = root,dir,file
	file_paths = []
	for i in file:
		a = os.path.join (root,i)
		file_paths.append (a)
	return file_paths


def get_columns_single (file_paths):
	"""
	得到所有测点名的交集
	"""
	result = []
	for path in file_paths:
		data_2 = pd.read_csv (path)  # ,encoding='gbk')
		result.append (set (data_2.columns.tolist ()))

	columns_single = result[0]
	for i in result[1:]:
		columns_single = columns_single & i
	return columns_single


def get_single_table (path,file_paths):
	"""
	得到单张数据表的数据
	"""
	data = pd.read_csv (path)  # ,encoding='gbk')
	temp_dataframe = pd.DataFrame ()
	columns_single = get_columns_single (file_paths)
	for column in columns_single:
		temp_dataframe[column] = data[column]
	return temp_dataframe


def get_all_corr_map (file_paths):
	"""
	绘制所有变量的相关性热力图
	"""
	result = []
	for path in file_paths:
		pdframe = get_single_table (path,file_paths)
		result.append (pdframe)
	total = pd.concat (result)
	plt.rcParams['axes.unicode_minus'] = False

	plt.figure (figsize=(30,30))
	plt.yticks (fontproperties='Times New Roman',size=25,weight='bold')  # 设置大小及加粗
	plt.xticks (fontproperties='Times New Roman',size=25,weight='bold')
	plt.imshow (np.corrcoef (total.T),'cool')  # ,#cmap='cool',alpha=1)
	cbar = plt.colorbar ()
	# cbar.set_label("热力值",fontsize=22)
	# cbar.formatter.set_scientific(True)
	# cbar.formatter.set_powerlimits((0,0))
	cbar.ax.tick_params (labelsize=40)  # 改变bar标签字体⼤⼩
	# cbar.ax.yaxis.get_offset_text().set_fontsize(18) #改变bar顶部字体⼤⼩
	cbar.update_ticks ()

	plt.title ('相关性热力图',fontsize=40)
	plt.xlabel ("参数",fontsize=40)
	plt.ylabel ("参数",fontsize=40)
	plt.savefig (r"D:\file_comb\研究生毕业论文\中期检查\ppt\图片\相关性.svg",dpi=300,bbox_inches='tight')
	plt.show ()


def get_special_data (data,wenyaqi=False):
	"""
	返回传入数据的各个参数
	"""
	if wenyaqi:
		result = pd.DataFrame ()
		# 过程量
		result["稳压器液位"] = (data["rcp008mn"]+data["rcp011mn"]+data["rcp007mn"])/3
		result["稳压器压力"] = (data['rcp037mp'] + data['rcp039mp']) / 2
		result["主蒸汽流量"] = (data['vvp001md'] + data['vvp002md'] + data['vvp003md'])/4000
		result["稳压器泄压箱压力"] = data["3rcp024mp_a_pnt"]

		# result["辅助给水流量(宽量程)"] = (data["asg002km"]+data["asg003km"]+data["asg001km"])/3
		# result["辅助给水流量(窄量程)"] = (data["asg012km"] + data["asg013km"] + data["asg011km"]) / 3


		# 控制量
		result["喷淋阀开度"] = (data["3rcp002vp_lc_obk"]+data["3rcp001vp_lc_obk"])/2
		result["蒸汽排放阀开度"] =  (data['3gct131vv_lc_obk'] + data['3gct132vv_lc_obk'] + data['3gct133vv_lc_obk']) / 3
		result["上冲流量调节阀开度"] = (data["3rcv046vp_lc_obk"])
		result["电加热器数量"] =sum([np.where(data[f"3rcp00{i}rs_p_pack"]>0,1,0) for i in range(1,7)])

	else:
		result = pd.DataFrame ()
		result["一回路压力平均值"] = (data['rcp037mp'] + data['rcp039mp']) / 2
		result["一回路温度变化速率"] = (data['temp_speed'])
		result["堆芯出口温度平均值"] = (data['3ric100km'] + data['3ric200km']) / 2
		result["主蒸汽压力平均值"] = (data['vvp007mp'] + data['vvp010mp'] + data['vvp013mp']) / 3
		# 负相关
		result['辅助给水流量平均值'] = (data['asg012km'] + data['asg013km']) / 2
		result['重要厂用水系统压力'] = data['sec004mp']
		result['DeltaTsat平均值'] = (data['3ric103km'] + data['3ric203km']) / 2
		result['安全壳压力平均值'] = (data['ety103mp'] + data['ety104mp']) / 2
		result['SG1水位平均值'] = (data['are052mn'] + data['are010mn'] + data['are055mn'] + data['are058mn']) / 4
		result['稳压器泄压箱压力'] = data['3rcp024mp_a_pnt']
		result['压缩空气系统压力'] = data['sap001mp']
		# 动作
		#result["汽机旁路系统蒸汽阀平均开度"] = (data['3gct131vv_lc_opnt'] + data['3gct132vv_lc_opnt'] + data['3gct133vv_lc_opnt']) / 3
		# result["一回路温度变化速率"] = (data['temp_speed'])
		# result["下泄流量"] = (data['rcp006mp'])
		# result["主给水流量"] = (data['rcv004mp'])
		# result["SG1蒸汽流量"] = (data['are046md'])
		#
		# # 负相关
		# result['辅助给水流量平均值'] = (data['asg012km'] + data['asg013km']+data['asg002km'] + data['asg003km']) / 4
		# result['电加热数量'] = sum([np.where(data[f"3rcp00{i}rs_p_pack"]>0,1,0) for i in range(1,7)])
		# result['重要厂用水系统压力'] = data['sec004mp']
		# 动作
		result["汽机旁路系统蒸汽阀平均开度"] = (data['3gct131vv_lc_opnt'] + data['3gct132vv_lc_opnt'] + data['3gct133vv_lc_opnt']) / 3
	return result

def get_target_data(data,sort:bool):
	new_pdframe = get_special_data (data,wenyaqi=sort)
	return new_pdframe



def get_all_data (file_paths,final_feature = False):
	"""
	返回列表，包含所有数据表的特征和奖励
	"""
	result = []
	for path in file_paths:
		pdframe = get_single_table (path,file_paths)
		pdframe.convert_dtypes ()
		if final_feature:
			new_pdframe = pdframe[final_feature[0:-2]]
			new_pdframe["first_loop_average_tmp"] = (new_pdframe['3rcp028mt'] + new_pdframe['3rcp029mt']) / 2
			first_loop_average_tmp = new_pdframe['first_loop_average_tmp'].tolist ()
			temp_change_speed = []
			last_tmp = first_loop_average_tmp[0]
			for i in first_loop_average_tmp:
				speed = (i - last_tmp) * 3600
				temp_change_speed.append (speed)
				last_tmp = i
			new_pdframe["temp_change_speed"] = temp_change_speed
			temp_change_speed = new_pdframe["temp_change_speed"].tolist ()
			rewards = []
			for i in temp_change_speed:
				reward = 100 - abs (i - (-56))
				rewards.append (reward)
			new_pdframe["action_average"] = (new_pdframe['3gct131vv_lc_opnt'] + new_pdframe['3gct132vv_lc_opnt'] +
			                                 new_pdframe['3gct133vv_lc_opnt']) / 3
			new_pdframe["reward"] = rewards
			result.append (new_pdframe)
		else:
			new_pdframe = get_special_data (pdframe)
			first_loop_average_tmp = new_pdframe['一环路冷热段平均值'].tolist ()
			temp_change_speed = []
			last_tmp = first_loop_average_tmp[0]
			for i in first_loop_average_tmp:
				speed = (i - last_tmp) * 3600
				temp_change_speed.append (speed)
				last_tmp = i
			rewards = []
			for i in temp_change_speed:
				reward = 100 - abs (i - (-56))
				rewards.append (reward)
			new_pdframe["动作"] = new_pdframe["汽机旁路系统蒸汽阀平均开度"]
			new_pdframe["奖励"] = rewards

			result.append (new_pdframe)

	return result


def offset_act_rew (data):
	"""
	偏移数据表的动作和奖励
	"""
	results = []
	for i in range (len (data)):
		action = data[i]["动作"].to_list ()
		action.pop (0)
		action.append (0)
		data[i]["动作"].iloc[:] = action
		reward = data[i]["奖励"].to_list ()
		reward.pop (0)
		reward.append (0)
		data[i]["奖励"].iloc[:] = reward
		result = data[i].drop (index=(len (data[i]) - 1))
		results.append (result)
	return results


# def get_normalized_dataframe (total_data_frame1):
# 	# 返回传入数据表的标准化形式
# 	std_scaler = preprocessing.StandardScaler ().fit (total_data_frame1)
# 	total_data_frame2 = std_scaler.transform (total_data_frame1)
# 	total_data_frame2 = pd.DataFrame (total_data_frame2,columns=total_data_frame1.columns)
# 	return total_data_frame2


def get_start_end (pdframe):
	"""
	返回传入数据表列表的开头和结尾位置
	"""
	len_result = []
	for i in pdframe:
		len_result.append (len (i))

	start,end = 0,0
	result = []
	for num,i in enumerate (len_result):
		start = end
		end = start + i
		result.append ((start,end))
	return result


def get_sars_total_normalized_data (offset_data,start_end):
	"""
	得到标准化后的sars元组数据
	pd_results：标准化后的各个数据表的sars
	total_len：标准化后总数据长度
	features_len：特征个数
	"""
	total_data_frame = pd.concat (offset_data,ignore_index=True)
	std_data = get_normalized_dataframe (total_data_frame)
	total_len = len (std_data)
	total_feature_len = len (total_data_frame.columns)
	features_len = total_feature_len - 2
	pd_results = []
	for start,end in start_end:
		result = []
		for row in range (start,end):
			sars = {}
			rows_data = std_data.iloc[row,:].to_list ()
			s = rows_data[0:features_len]
			a = rows_data[features_len]
			r = rows_data[total_feature_len - 1]
			if row < end - 1:
				s_ = std_data.iloc[row + 1,:].to_list ()[0:features_len]
				sars["s"],sars["a"],sars["r"],sars["s_"] = s,a,r,s_
				result.append (sars)
		pd_results.append (result)
	return pd_results,total_len,features_len


def get_action_minmax (offset_data):
	"""
	返回动作的区间
	"""
	total_data_frame = pd.concat (offset_data,ignore_index=True)
	std_data = get_normalized_dataframe (total_data_frame)
	action_max = max (std_data["动作"].values)
	action_min = min (std_data["动作"].values)
	return action_max,action_min


#file_paths = get_csv (r"E:\file_comb\核二院\数据\wuhan\数据采集0421\SGTR")
# final_feature=pd.read_csv("final_feature.csv")["0"].tolist()
def get_all_data2 (file_paths,final_feature = False):
	"""
	返回包含所有数据表的列表，不包含特征和奖励
	"""
	result = []
	for path in file_paths:
		pdframe = get_single_table (path,file_paths)
		pdframe.convert_dtypes ()
		new_pdframe = get_special_data (pdframe)

		result.append (new_pdframe)

	return result

# alldata_list = get_all_data2 (file_paths)
#
# start_end = get_start_end (alldata_list)

def get_total_normalized_data (data,start_end):
	"""
	得到标准化后的数据
	action_max,action_min：action的极值
	total_feature_len：特征个数
	"""
	total_data_frame = pd.concat (data,ignore_index=True)
	std_data = get_normalized_dataframe (total_data_frame)
	action_max = max (std_data["汽机旁路系统蒸汽阀平均开度"].values)
	action_min = min (std_data["汽机旁路系统蒸汽阀平均开度"].values)
	total_len = len (std_data)
	total_feature_len = len (total_data_frame.columns)

	pd_results = []
	for start,end in start_end:
		result = std_data.iloc[start:end,:]
		pd_results.append (result)
	return pd_results,action_max,action_min,total_feature_len


# total_normalized_data,action_max,action_min,total_feature_len = get_total_normalized_data (alldata_list,start_end)
# aaa={"action_max":action_max,"action_min":action_min}
# #action_max,action_min = get_action_minmax (alldata_list)
# total_normalized_data.append(aaa)
#
#dataa=np.array(alldata_list,dtype='object')
# np.save ("全部标准化数据列表",dataa)
