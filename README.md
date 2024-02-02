# HRICA
[ICRA'24] Human-Robot Interactive Creation of Artistic Portrait Drawings

人机协同创作艺术画像：
- 1 个新的数据集 CelebLine：利用 AiSketcher + Simplify 将 CelebAMask-HQ 转换为新的线条画数据集
- 1 个新的画像补全算法 GAPDI：Mask-Free 和 结构感知的图像补全算法
- 1 个新的人机系统创作系统 HRICA：人和机器人，协同操作界面，ControlNet扩展

## Paper Information

Fei Gao, Lingna Dai, Jingjie Zhu, Mei Du, Yiyuan Zhang, Maoying Qiao, Chenghao Xia, Nannan Wang, and Peng Li \*，
Human-Robot Interactive Creation of Artistic Portrait Drawings, 
2024 IEEE International Conference on Robotics and Automation (ICRA), accepted, May13-17, 2024, Yokohama, Japan. 
(\* Corresponding Author)

> Xidian University (西安电子科技大学); AiSektcher Technology (杭州妙绘科技有限公司); Hangzhou Dianzi University (杭州电子科技大学);
The University of Technology, Sydney (悉尼科技大学); The University of Sydney (悉尼大学); 
Institute of Software, Chinese Academy of Sciences (中科院软件所); University of Chinese Academy of Sciences Beijing/Nanjing/Hangzhou (中国科学院大学) and Nanjing Institute of Software Technology (南京软件研究院). 

## Abstract
In this paper, we present a novel system for *Human-Robot Interactive Creation of Artworks* (HRICA). Different from previous robot painters, HRICA allows a human user and a robot to alternately draw strokes on a canvas, to collaboratively create a portrait drawing through frequent interactions. The key is to enable the robot to understand human intentions, during the interactive creation process.We here formulate this as a mask-free image inpainting problem, and propose a novel method to estimate the complete version of a portrait drawing, after the human user has drawn some initial strokes. In this way, the robot can select some complementary strokes and draw them on the canvas. To train and evaluate our inpainting method, we construct a novel large-scale portrait drawing dataset, CelebLine, which composes of high-quality portrait line-drawings, with dense labels of both 2D semantic parsing masks and 3D depth maps. Finally, we develop a human-robot interactive drawing system with low-cost hardware, user-friendly interface, and interesting creation experience. Experiments show that our robot can stably cooperate with human users to create diverse styles of portrait drawings. In addition, our portrait drawing inpainting method significantly outperforms previous advanced methods. We will release the code and dataset after peer review.


