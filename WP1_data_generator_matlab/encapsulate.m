function [A] = encapsulate(X, window)
    shape = size(X);
    length = shape (2);
   
    

    
    A = [];
 %   B = [];
    for i=1:(length-window)
        tempWindowX = X(:,i:i+(window-1));
        A = cat(3, A, tempWindowX);  %[A, tempWindowX]; 

 %       tempWindowY = Y(:,i:i+(window-1));
 %       B = cat(3, B, tempWindowY);  %[B, tempWindowY]; 
    end
end