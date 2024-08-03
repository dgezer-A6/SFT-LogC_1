#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import pandas
import numpy
from rdkit.Chem import Descriptors
from rdkit.Chem import Descriptors3D
from rdkit import Chem 
from rdkit.Chem import AllChem
from math import log, log10
import matplotlib.pyplot as plt


# In[ ]:


dataset= pandas.read_excel("Your File Path")   # Write the file name or file path

features_2d = []
features_3d = []

for i in dataset["SMILES"]:
    mol = Chem.MolToSmiles(Chem.MolFromSmiles(i))
    mol = Chem.MolFromSmiles(mol)
    mol_2=Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_2, randomSeed = 511)
    if AllChem.EmbedMolecule(mol_2, randomSeed = 511) == -1:
        AllChem.EmbedMolecule(mol_2,useRandomCoords=True, randomSeed = 511)
    AllChem.MMFFOptimizeMolecule(mol_2)
    descriptor_values_2d = Descriptors.CalcMolDescriptors(mol).values()
    descriptor_values_3d = Descriptors3D.CalcMolDescriptors3D(mol_2).values()
    features_2d = features_2d + [list(descriptor_values_2d)]
    features_3d = features_3d + [list(descriptor_values_3d)]


# In[ ]:


descriptor_names_2d = [i[0] for i in Descriptors._descList ]
descriptor_names_3d = ['PMI1', 'PMI2', 'PMI3', 'NPR1', 'NPR2', 'RadiusOfGyration', 'InertialShapeFactor', 'Eccentricity', 'Asphericity', 'SpherocityIndex', 'PBF']
features_2d = pandas.DataFrame(features_2d, columns = descriptor_names_2d, index = dataset["Name"])
features_3d = pandas.DataFrame(features_3d, columns = descriptor_names_3d, index = dataset["Name"])
features = pandas.concat([features_2d, features_3d], axis = 1)


# In[ ]:


descriptors_shap_rfr_cmc = ['FpDensityMorgan3', 'BCUT2D_MRLOW', 'AvgIpc', 'PEOE_VSA6', 'SMR_VSA5', 'VSA_EState1', 'VSA_EState8', 'MolLogP', 'fr_Al_OH_noTert', 'PMI2'] 


# In[ ]:


descriptors_shap_rfr_kl = ['MinAbsEStateIndex', 'qed', 'FpDensityMorgan1', 'BCUT2D_MWLOW', 'BCUT2D_CHGLO', 'Chi1n', 'PEOE_VSA3', 'PEOE_VSA6', 'SMR_VSA5', 'VSA_EState1', 'VSA_EState8', 'MolLogP', 'fr_Al_OH_noTert', 'PMI2'] 


# In[ ]:


descriptors_shap_rfr_maxsec = ['qed', 'FpDensityMorgan2', 'BCUT2D_MWLOW', 'AvgIpc', 'PEOE_VSA3', 'SlogP_VSA2', 'SlogP_VSA8', 'EState_VSA2', 'fr_COO2', 'fr_C_O', 'PMI1', 'PMI2', 'InertialShapeFactor', 'PBF']


# In[ ]:


selected_features_list_cmc = descriptors_shap_rfr_cmc
selected_features_list_kl = descriptors_shap_rfr_kl
selected_features_list_maxsec = descriptors_shap_rfr_maxsec


# In[ ]:


filename_cmc = "model_shap_rfr_cmc_y"
filename_kl = "model_shap_rfr_kl_y"
filename_maxsec = "model_shap_rfr_maxsec_y"


# In[ ]:


results_list_cmc = []
results_list_kl = []
results_list_maxsec = []
features_cmc = features[selected_features_list_cmc]
features_kl = features[selected_features_list_kl]
features_maxsec = features[selected_features_list_maxsec]
with open("C:/Users/user/Saved_Models/" + filename_cmc, 'rb') as file:
    model = pickle.load(file)
result_cmc = model.predict(features_cmc)
results_list_cmc.append(list(result_cmc))
with open("C:/Users/user/Saved_Models/" + filename_kl, 'rb') as file:
    model = pickle.load(file)
result_kl = model.predict(features_kl)
results_list_kl.append(list(result_kl))
with open("C:/Users/user/Saved_Models/" + filename_maxsec, 'rb') as file:
    model = pickle.load(file)
result_maxsec = model.predict(features_maxsec)
results_list_maxsec.append(list(result_maxsec))


# In[ ]:


results_cmc = pandas.DataFrame(list(results_list_cmc),columns = dataset["Name"]).transpose()
results_cmc.columns = [filename_cmc]
results_kl = pandas.DataFrame(list(results_list_kl),columns = dataset["Name"]).transpose()
results_kl.columns = [filename_kl]
results_maxsec = pandas.DataFrame(list(results_list_maxsec),columns = dataset["Name"]).transpose()
results_maxsec.columns = [filename_maxsec]


# In[ ]:


R = 8.314
T = 25 + 273
sft_0 = 72 
#sft = sft_0 - R * T * gamma_max * log (1 + KL * c)


# In[ ]:


# Single Figure for All Molecules
for i in range(0,len(dataset.index)):
    sft = []
    log_c = []
    for c in numpy.arange(0.000000000000001,10**(result_cmc[i]),0.0000001):
        sft.append((sft_0 - R * T * result_maxsec[i]*(10**(-6))*(10**3) * log(1 + 1000*(10**result_kl[i]) * c)))
    sft.append(sft_0 - R * T * result_maxsec[i]*(10**(-6))*(10**3) * log(1 + 1000*(10**result_kl[i]) * (10**result_cmc[i])))
    conc = numpy.arange(0.000000000000001,10**(result_cmc[i]),0.0000001)
    log_c =[log10(i) for i in conc]
    log_c.append(0)
    plt.plot(log_c, sft)
plt.legend(dataset["Name"], loc="lower left")
plt.ylabel("SFT (mN/m)")
plt.xlabel("Log(C)")
plt.xlim(-15,0)
plt.ylim(0,80)
plt.show()


# In[ ]:


# A Separate Figure for Each Molecule
for i in range(0,len(dataset.index)):
    sft = []
    log_c = []
    for c in numpy.arange(0.000000000000001,10**(result_cmc[i]),0.0000001):
        sft.append((sft_0 - R * T * result_maxsec[i]*(10**(-6))*(10**3) * log(1 + 1000*(10**result_kl[i]) * c)))
    sft.append(sft_0 - R * T * result_maxsec[i]*(10**(-6))*(10**3) * log(1 + 1000*(10**result_kl[i]) * (10**result_cmc[i])))
    conc = numpy.arange(0.000000000000001,10**(result_cmc[i]),0.0000001)
    log_c =[log10(i) for i in conc]
    log_c.append(0)
    plt.plot(log_c, sft)
    plt.title(dataset["Name"][i])
    plt.ylabel("SFT (mN/m)")
    plt.xlabel("Log(C)")
    plt.xlim(-15,0)
    plt.ylim(0,80)
    plt.show()


# In[ ]:




