classdef ssimRegressionLayerHere < nnet.layer.RegressionLayer   & nnet.layer.Acceleratable
    methods
        function layer = ssimRegressionLayerHere(name)
            layer.Name = name;
            layer.Description = 'ssim';
   
        end
	 
       
        
        function loss = forwardLoss(layer, Y, T)
       
  Values  = calculateSSIM(Y, T);    
  loss = mean(Values , 'all'); 
		 
  
        end
		 
    end
end
 
 

function ssimValues = calculateSSIM(Y, T)
	   
  Y = rescale(Y); %important
  T = rescale(T);   
  K1 =0.01;
  K2 =0.01;
  L  =1;	
	 
		 
  muY = mean(Y, 'all' );
  muT = mean(T,  'all');  
  
  varY = var(Y, 1, 'all');  
  varT = var(T, 1, 'all');		
  
  covYT = cov(Y(:), T(:), 1);
  covYT = covYT(1, 2);  
   
		 
  ssimValues =1;
  
  C1 = (K1 * L)^2;
  C2 = (K2 * L)^2;
  numerator = (2 * muY * muT + C1) * (2 * covYT + C2);
  denominator = (muY^2 + muT^2 + C1) * (varY + varT + C2);
  ssimValues = ssimValues *  numerator / denominator;
  	   
  
	
end
	  
 