///param img:输入图像
///param WeightsPath:权重文件路径
///param CfgPath:网络文件路径
int Classifier(Mat& img,string& WeightsPath,string& CfgPath)
{
	cv::dnn::ClassificationModel ClassNet(WeightsPath, CfgPath);///定义分类模型对象
	if (ClassNet.empty())
	{
		cout << "Can't load the net" << endl;
		return -1;
	}
	////GPU计算
	//net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
	//net.setPreferableTarget(dnn::DNN_TARGET_CUDA_FP16);	
	///CPU
	ClassNet.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
	ClassNet.setPreferableTarget(dnn::DNN_TARGET_CPU);
	ClassNet.setInputSize(256, 256);///预处理，输入图像缩放
	int id = 0;///类别号
	float c = 0.0;///识别概率
	ClassNet.classify(img, id, c);
}
