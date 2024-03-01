# 中国健康成人超声心动图测量值生理性变异校正的数学方法

#### 注：本项目为国家级大学生创新创业训练计划项目，由中国海洋大学曾雪迎副教授指导，团队成员共有五人。队长：李璇，队员：朱甲文 宋晓菲 孙菊颐 王静。

#### 项目报告：[中国健康成人超声心动图测量值生理性变异校正的数学方法.pdf](https://github.com/andone-07/EMINCA-physiologic-variation-correction-based-on-mathematical-method-and-MLP/blob/master/%E4%B8%AD%E5%9B%BD%E5%81%A5%E5%BA%B7%E6%88%90%E4%BA%BA%E8%B6%85%E5%A3%B0%E5%BF%83%E5%8A%A8%E5%9B%BE%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90%E4%B8%8E%E7%94%9F%E7%90%86%E6%80%A7%E5%8F%98%E5%BC%82%E7%9F%AB%E6%AD%A3%E7%9A%84%E6%95%B0%E5%AD%A6%E6%96%B9%E6%B3%95.pdf)
—————————————————————————————————————————————

### 项目介绍

超声心动图参数正常值是判断心脏结构与功能正常与否的重要依据，健康成人超声心动图参数测值在不同性别、年龄和种族之间存在显著性差异。长期以来，我国临床上采用的超声心动图参考值主要采用欧美标准，由于种族和体型的巨大差异，欧美标准并不适合于中国人群。中国缺乏大样本、多中心健康成人的超声心动图参考值，因此，亟待建立具有广泛代表性、大样本的超声心动图正常参考值。在中国工程院张运院士的倡导下，由中华医学会超声医学分会设计、组织并实施的一项前瞻性、全国多中心的《中国汉族健康成年人超声心动图测值研究》（EMINCA研究），历时3年余，全国43家三甲医院参加，共纳入1394例、年龄18-79岁的汉族健康成人志愿者，按照ASE/EACVI国际超声心动图指南标准进行图像采集和数据测量，建立了中国汉族健康成年人超声心动图参数的大样本数据库，共获得34个心腔与大动脉的二维参数和37个血流与组织多普勒参数的测值，是迄今为止世界上样本量最大、测量参数最多、代表性最强的超声心动图正常值研究，从而确立了真正意义上的“中国标准”。

EMINCA研究及国外同类研究结果显示，超声心动图参数正常值受性别、年龄和体型等生物学特征变量的显著影响，如果忽视不同性别、年龄和体型对超声心动图测值所造成的生理性差异，全部采用一个标准，必将导致心腔大小及心脏功能正常与否的误判和误诊。因此，临床上需要对超声心动图参数测值的生理性变异进行科学合理的校正，以消除个体间因年龄、性别、体型的差异对测值所产生的生理性影响，有利于不同个体间、群体间进行合理的比较、以及正常与异常的判断。长期以来，超声心动图参数的校正方法学一直是国际超声影像学领域研究的热点，更是困扰该领域长达半个多世纪的科学难题。

齐鲁医院姚桂华教授、中国海洋大学曾雪迎副教授等在张运院士的指导下，通过对EMINCA研究数据库以及国外同类研究结果进行深入分析与探讨后，发现不同的超声心动图参数与不同的生物学特征变量（包括年龄、性别、身高、体重等）中的一个或多个变量呈不同的非线性相关关系，在试验多个数学模型之后，科学大胆地提出了优化的多变量非等距模型，最终成功地对34个二维超声心动图参数进行了多变量非线性校正，消除了年龄、性别、身高、体重所产生的生理性影响，验证该模型的校正成功率达100%。相关研究论文在国际超声影像学领域的顶级杂志《Journal of the American Society of Echocardiography》上发表，不仅表明超声心动图正常值的“中国标准”得到了国内外同行的广泛肯定，而且成功解决了长期困扰血管影像学领域的关于超声心动图测值生理性变异校正的方法学难题，填补了该研究领域的国内外空白，对目前在国内外超声心动图指南中推荐的、在临床实践中长期采用的体表面积线性校正法提出了强有力挑战，必将对国际心血管超声影像学领域的临床研究和参考指南的制定产生积极和深远的影响。

**本项目的任务，是综合运用数学统计、神经网络与深度学习等方法，探寻年龄因素影响超声心动图各功能参数的客观规律，分析关键的年龄拐点，对临床检查的重点人群给出相应建议。基于多层感知机模型（MLP），用70%的数据进行训练，找到更为精确的回归方程，对超声心动图的生理性变异进行校正，即找到符合中国健康成人的超声心动图正常值的参考指标。对各指标之间进行分析，探索指标之间和指标组之间的联动关系或其他联系，从而为医学上的检查提供思路与建议。**

 
