# load the label of classe
ClasseFile = 'datasets.names'
classes = None
with open(ClasseFile, 'r') as f :
	#classes = len(f.readlines())
	classes = f.read().rstrip('\n').split('\n')
	print(classes)
	print('The number of classes is :{}'.format(len(classes)))