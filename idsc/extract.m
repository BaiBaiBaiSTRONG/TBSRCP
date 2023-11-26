addpath common_innerdist;
clear;

%------ Parameters ----------------------------------------------
ifig		= 1;
sIms		= {};
sImn		= {};
save_path   = './saved_shape_context/';

%-- shape context parameters
n_dist		= ;
n_theta		= ;
bTangent	= 1;
bSmoothCont	= 1;
n_contsamp	= ;

%-- Extract inner-distance shape context
figure(ifig);	clf; hold on;	set(ifig,'color','w'); colormap(gray);
for k=1:length(sIms)
	
	%- Contour extraction
	ims	= double(imread(sIms{k}));
    ims = ims(:, :, 1)
	Cs	= extract_longest_cont(ims, n_contsamp);
	
	%- inner-dist shape context
	msk		= ims;%>.5;
	[sc,V,E,dis_mat,ang_mat] = compu_contour_innerdist_SC( ...
									Cs,msk, ...
									n_dist, n_theta, bTangent, bSmoothCont,...
									0);
    file_name = strcat(save_path, sImn{k});
    file_name = strcat(file_name, '.png')
    imwrite(sc, file_name);
	
end

