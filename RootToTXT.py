from root_numpy import root2array, rec2array
import numpy as np

def BJetRootToTXT(infile,outfile):
	branch_names_= """Jet_genjetPt, Jet_pt, Jet_eta, Jet_mt, Jet_leadTrackPt, Jet_leptonPtRel_new, Jet_leptonPt, Jet_leptonDeltaR, Jet_neHEF, Jet_neEmEF, Jet_vtxPt, Jet_vtxMass, Jet_vtx3dL, Jet_vtxNtrk, Jet_vtx3deL_new, nGoodPVs, Jet_PFMET, Jet_METDPhi, Jet_JetDR"""
	branch_names=branch_names_.split(",")
	branch_names= [c.strip() for c in branch_names] #delete space in the string
	data= root2array(infile,"jet",branch_names,start=1,stop=4000000) #total: 11137276
	np.savetxt(outfile,data,header=branch_names_.replace(",",""),comments='')
	
BJetRootToTXT("/data7/cyeh/Summer16_BjReg/TrainingTree/minitree_4b_2_26.root","/data7/cyeh/Summer16_BjReg/TrainingTree/minitree_4b_2_26.txt")	
BJetRootToTXT("/data7/cyeh/Summer16_BjReg/TrainingTree/minitree_4b_leading_2_26.root","/data7/cyeh/Summer16_BjReg/TrainingTree/minitree_4b_leading_2_26.txt")
BJetRootToTXT("/data7/cyeh/Summer16_BjReg/TrainingTree/minitree_4b_trailing_2_26.root","/data7/cyeh/Summer16_BjReg/TrainingTree/minitree_4b_trailing_2_26.txt")	