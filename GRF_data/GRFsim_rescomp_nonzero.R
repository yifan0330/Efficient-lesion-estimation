# Reinstall oro.nifti for current R version
if (!require(oro.nifti, quietly=TRUE)) {
  install.packages("oro.nifti", repos="http://cran.r-project.org", lib="/gpfs3/users/nichols/pra123/R/x86_64-pc-linux-gnu-library/4.3/")
}
library(oro.nifti, lib.loc="/gpfs3/users/nichols/pra123/R/x86_64-pc-linux-gnu-library/4.3/")

# install.packages("neurobase", repos="http://cran.r-project.org", lib="/gpfs3/users/nichols/pra123/R/x86_64-pc-linux-gnu-library/4.3/")
library(neurobase, lib.loc="/gpfs3/users/nichols/pra123/R/x86_64-pc-linux-gnu-library/4.3/")


# install.packages("sp", repos="http://cran.r-project.org", lib="/gpfs3/users/nichols/pra123/R/x86_64-pc-linux-gnu-library/4.3/")
library(sp, lib.loc="/gpfs3/users/nichols/pra123/R/x86_64-pc-linux-gnu-library/4.3/")

# Install RandomFieldsUtils from archive
# install.packages("https://cran.r-project.org/src/contrib/Archive/RandomFieldsUtils/RandomFieldsUtils_1.2.5.tar.gz", 
                # repos=NULL, type="source", lib="/gpfs3/users/nichols/pra123/R/x86_64-pc-linux-gnu-library/4.3/")
library(RandomFieldsUtils, lib.loc="/gpfs3/users/nichols/pra123/R/x86_64-pc-linux-gnu-library/4.3/")

# Install RandomFields from archive  
# install.packages("https://cran.r-project.org/src/contrib/Archive/RandomFields/RandomFields_3.3.14.tar.gz", 
#                 repos=NULL, type="source", lib="/gpfs3/users/nichols/pra123/R/x86_64-pc-linux-gnu-library/4.3/")
library(RandomFields, lib.loc="/gpfs3/users/nichols/pra123/R/x86_64-pc-linux-gnu-library/4.3/")


# install.packages("fslr", repos="http://cran.r-project.org", lib="/gpfs3/users/nichols/pra123/R/x86_64-pc-linux-gnu-library/4.3/")
library(fslr, lib.loc="/gpfs3/users/nichols/pra123/R/x86_64-pc-linux-gnu-library/4.3/")


#load the brain mask
brain_mask = readNIfTI("/well/nichols/users/pra123/brain_lesion_project/GRF_new_data/MNI152_T1_2mm_brain_mask.nii.gz")
empir_prob = readNIfTI("/well/nichols/users/pra123/brain_lesion_project/GRF_new_data/empir_prob_mask.nii.gz")
#-----------------------------------------------------------------------------------------------------------
#read in arguments from command line
#!/usr/bin/env Rscript
Nargs = 5
args = commandArgs(trailingOnly=TRUE)

if (length(args)!=Nargs) {
  stop("Wrong number of arguments.")
}

var = as.numeric(args[1])
scale = as.numeric(args[2])
n_subj = as.numeric(args[3])
workdir = args[4]
seed = as.numeric(args[5])
#-----------------------------------------------------------------------------------------------------------

#for look to try different ages, write down number of lesion voxels and save one slice each

model = RMgauss(var=var, scale=scale)
x_lim = 91
y_lim = 109
z_lim = 91
x.seq = 1:x_lim
y.seq = 1:y_lim
z.seq = 1:z_lim



set.seed(1)
ages = round(runif(n_subj, min=45.12, max=80.65),digits=2)
ages = sort(ages)
print("Simulated data mean age")
mean(ages)

#fix the UKBB results used to create the age effect
coef_age_ukbb = readNIfTI("/well/nichols/users/pra123/brain_lesion_project/GRF_new_data/coef_age_nvars_1_method_2.nii.gz")
coef_intercept_ukbb = readNIfTI("/well/nichols/users/pra123/brain_lesion_project/GRF_new_data/coef_(Intercept)_nvars_1_method_2.nii.gz")
print("UKBB mean age")
mean(coef_age_ukbb)

##create a data set of those subjects
nsubj_dataframe = data.frame(id = 1:n_subj, age = ages)
nsubj_dataframe$file_name = ""

## added by PK on 28th Feb 2020
nsubj_dataframe$age = scale(nsubj_dataframe$age, scale=F)

#setwd(workdir)
set.seed(seed)

for (i in 1:n_subj){
  print(i)
  filename = paste0(i,"_T2_lesions_MNI_bin.nii.gz")
  nsubj_dataframe$file_name[i] = filename
  
  sim = RFsimulate(model, x = x.seq, y = y.seq, z.seq, grid=T, spConform=FALSE)
  
  sim_nifti =as.nifti(sim)
  dim(sim_nifti)
  
  grf_img = sim_nifti
  #grf_img[brain_mask==0] = 0 # exclude non-brain tissue
  #grf_img[empir_prob==0] = 0 # exclude zero-lesion voxels
  #filename = paste0(workdir, "/", i,"_T2_lesions_MNI_grf")
  #writeNIfTI(grf_img, filename)
  
  xbeta = coef_intercept_ukbb
  #xbeta = coef_intercept_ukbb + nsubj_dataframe$age[i] * coef_age_ukbb
  #xbeta[empir_prob==0] = 0 # exclude zero-lesion voxels
  # filename=paste0("plots_temp/xbeta_z45_subj",i, "_age",ages[i],".pdf")
  # pdf(filename)
  # image(xbeta, z=45, plot.type="single")
  # dev.off()
  #filename = paste0(workdir, "/", i,"_T2_lesions_MNI_xbeta")
  #writeNIfTI(xbeta, filename)
 
  z = pnorm(grf_img + xbeta)
  #z[empir_prob==0] = 0 # exclude zero-lesion voxels 
  # filename=paste0("plots_temp/y_cont_z45_subj",i, "_age",ages[i],".pdf")
  # pdf(filename)
  # image(y_cont, z=45, plot.type="single")
  # dev.off()
  #filename = paste0(workdir, "/", i,"_T2_lesions_MNI_z")
  #writeNIfTI(z, filename)
  
  y_bin = z
  y_bin[z<0.5] = 0
  y_bin[z>=0.5] = 1
  y_bin[brain_mask==0] = 0
  y_bin[empir_prob==0] = 0
  y_bin = as.nifti(y_bin)
  # filename=paste0("plots_sample500/y_bin_z45_subj",i, "_age",ages[i],".pdf")
  # pdf(filename)
  # image(y_bin, z=45, plot.type="single")
  # dev.off()
  filename = paste0(workdir, "/", i,"_T2_lesions_MNI_bin")
  writeNIfTI(y_bin, filename)
  
  # clustered_map = fsl_cluster(file = y_bin, threshold = 0.5, connectivity = 6, mm = T)
  # filename = paste0(i,"_T2_lesions_MNI_cluster.txt")
  # write.table(clustered_map,
  #             file=filename,
  #             row.names=FALSE, sep="\t", quote=FALSE)
  # nsubj_dataframe$lesion_num[i] = clustered_map[1,1]
  # nsubj_dataframe$lesion_vol[i] = sum(clustered_map[,2])
  # nsubj_dataframe$lesion_avg_vol[i] = nsubj_dataframe$lesion_vol[i] / nsubj_dataframe$lesion_num[i]
  
  #pnorm_z = pnorm(z)
  #pnorm_z[empir_prob==0] = 0
  # filename=paste0("plots_sample500/pnorm_y_cont_z45_subj",i, "_age",ages[i],".pdf")
  # pdf(filename)
  # image(pnorm_y_cont[[i]], z=45, plot.type="single")
  # dev.off()
  # 
  # intensity_plot(slice=45,ncols=256, cols=cols2, img=pnorm_y_cont[[i]], mask=brain_mask, name=filename,legend=T, min=0, max=1)
  #filename = paste0(workdir, "/", i,"_T2_lesions_MNI_pnormZ")
  #writeNIfTI(pnorm_z, filename)
  
  if(i%%100==0) {print(i)}
   
  if(i==1) {
    empir_temp = y_bin
  } else {
    empir_temp = empir_temp + y_bin 
  }
  
  if(i%%4==0) {
    if(i==4) {
    empir_temp_quarter = y_bin
    } else {
      empir_temp_quarter = empir_temp_quarter + y_bin
    }
  }

}

#-------------------------------------------------------------------
# save the empirical probability masks as part of the simulation step
empir_temp = empir_temp / n_subj
filename = paste0(workdir, "/empir_prob_mask_", n_subj, "subj")
writeNIfTI(empir_temp, filename)

empir_temp_quarter = empir_temp_quarter / (n_subj/4)
filename = paste0(workdir, "/empir_prob_mask_", n_subj/4, "subj")
writeNIfTI(empir_temp_quarter, filename)

write.table(nsubj_dataframe[,1:3],
            file=paste0(workdir, "/GLM_sample", n_subj, ".dat"),
            row.names=FALSE, sep="\t", quote=FALSE)

nsubj_dataframe$Intercept = rep(1, n_subj)
#colnames(nsubj_dataframe)
nsubj_dataframe = nsubj_dataframe[,colnames(nsubj_dataframe)[c(1,4,2,3)]]

write.table(nsubj_dataframe,
            file=paste0(workdir, "/BSGLMM_sample", n_subj, ".dat"),
            row.names=FALSE, sep="\t", quote=FALSE)

nsubj_dataframe$age = ages # raw ages
write.table(nsubj_dataframe[,colnames(nsubj_dataframe)[c(1,3,4)]],
            file=paste0(workdir, "/sample", n_subj, ".dat"),
            row.names=FALSE, sep="\t", quote=FALSE)

nsubj_dataframe = nsubj_dataframe[(1:n_subj)%%4==0,]
write.table(nsubj_dataframe[,colnames(nsubj_dataframe)[c(1,3,4)]],
            file=paste0(workdir, "/", "sample", n_subj/4, ".dat"),
            row.names=FALSE, sep="\t", quote=FALSE)

nsubj_dataframe$age = scale(nsubj_dataframe$age, scale=F)
write.table(nsubj_dataframe[,colnames(nsubj_dataframe)[c(1,3,4)]],
            file=paste0(workdir, "/GLM_sample", n_subj/4, ".dat"),
            row.names=FALSE, sep="\t", quote=FALSE)

write.table(nsubj_dataframe,
            file=paste0(workdir, "/BSGLMM_sample", n_subj/4, ".dat"),
            row.names=FALSE, sep="\t", quote=FALSE)


