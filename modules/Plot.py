# import pandas
import matplotlib.pyplot as plt
import numpy as np

def compare_regressions(decisions, Labels, XLabel, bins=30, min_=None, max_=None, norm=True, Xlimit=False, outfilename = None):
	if (min_ is None) and (max_ is None) :
		low = min(np.min(d) for d in decisions)
		high = max(np.max(d) for d in decisions)
		low_high = (low,high)
		print('min: '+ str(low)+', max: '+str(high))
	else:
		low_high = (min_,max_)
	
	colors = 'r','b','g','m'
	fig = plt.figure()
	# plt.subplots()
	for nPl in range(len(decisions)):
		if Xlimit != False:
			plt.hist(decisions[nPl],
					 color=colors[nPl], alpha=0.5, bins=bins,
					 histtype='stepfilled', density=norm,
					 label=Labels[nPl])
		else:
			plt.hist(decisions[nPl],
					 color=colors[nPl], alpha=0.5, range=low_high, bins=bins,
					 histtype='stepfilled', density=norm,
					 label=Labels[nPl])
	# hist, bins = np.histogram(decisions[1],
							  # bins=bins, range=low_high, normed=True)
	# scale = len(decisions[1]) / sum(hist)
	# err = np.sqrt(hist * scale) / scale

	# width = (bins[1] - bins[0])
	# center = (bins[:-1] + bins[1:]) / 2
	# plt.errorbar(center, hist, yerr=err, fmt='o', c='r', label='truth')

	plt.xlabel(XLabel)
	if norm==True :
		plt.ylabel("Arbitrary units")
	else:
		plt.ylabel("Events")
		
	if Xlimit != False:
		plt.xlim(low_high)
	
	plt.legend(loc='best')
	if outfilename != None:
		fig.savefig(str(outfilename)+'.png')
	
def input_dstr(Datasets, Columns, XLabels, DLabels, outfilename='Input_Dtr', norm=True):
	print('#subplots:'+str(len(Columns)))
	colors = 'r','b','g','y'
	
	nFig = int((len(Columns)/9))
	if (len(Columns)%9) != 0:
		nFig=nFig+1
	for nfig in range(nFig):
		nXplots = 3
		nYplots = 3
		nplots = 9
		if nfig==(nFig-1):
			# print(len(Columns)%9)
			nplots = len(Columns)%9
			if (len(Columns)%9) != 0:
				if (len(Columns)%9) < 2:
					nXplots = 1
					nYplots = 1
				elif (len(Columns)%9) < 3:
					nXplots = 2
					nYplots = 1
				elif (len(Columns)%9) < 4:
					nXplots = 3
					nYplots = 1
				elif (len(Columns)%9) < 7:
					nXplots = 3
					nYplots = 2
				
		# print('nP:'+str(nXplots)+':'+str(nYplots))
		fig = plt.figure(figsize=(6*nXplots,5*nYplots))
		for nPl in range(nplots):
			ax = plt.subplot(nYplots, nXplots, nPl+1)
			# plt.subplots(ncols=nPl+1, figsize=(6,5))
			Hist = []
			for nDataset in range(len(Datasets)):
				Hist += plt.hist(Datasets[nDataset][:,Columns[nfig*9+nPl]],
							 color=colors[nDataset], alpha=0.5, bins=30,# range=low_high,
							 histtype='stepfilled', normed=norm,
							 label=DLabels[nDataset])
			
			plt.xlabel(XLabels[nfig*9+nPl])
			plt.legend(loc='best')
			
			low = min(np.min(Datasets[nDataset][:,Columns[nfig*9+nPl]]) for nDataset in range(len(Datasets)))
			high = max(np.max(Datasets[nDataset][:,Columns[nfig*9+nPl]]) for nDataset in range(len(Datasets)))
			hight = min(Hist[nDataset].max() for nDataset in range(len(Datasets)))
			# high = max(np.max(d) for d in decisions)
			ax.text(low+0.1*(high-low), hight*0.9, 'min: ' + str(low))
			ax.text(low+0.1*(high-low), hight*0.8, 'max: ' + str(high))
		fig.savefig(str(outfilename)+'_'+str(nfig)+'.png')


# def box(Datasets, Columns, XLabels, outfilename='Box'):
	# print('#subplots:'+str(len(Columns)))
	

def show():
	plt.tight_layout()
	plt.show()