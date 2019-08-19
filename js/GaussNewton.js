function Deriv(Func, input, params, n) {  
    // Assumes input is a single row matrix  
  
    // Returns the derivative of the nth parameter  
    var params1 = params.clone();  
    var params2 = params.clone();  
  
    // Use central difference  to get derivative  
    params1.set(n,0,params1.get(n,0)-DERIV_STEP);  
    params2.set(n,0,params2.get(n,0)+DERIV_STEP);  
  
    var p1 = Func(input, params1);  
    var p2 = Func(input, params2);  
  
    var d = (p2 - p1) / (2*DERIV_STEP);  
  
    return d;  
}  

function GaussNewton(Func, inputs, outputs, params) {  
    var m = inputs.shape[0];  
    var n = inputs.shape[1];  
    var num_params = params.shape[0];  
  
    var r=nj.zeros([m,1]); // residual matrix  
    var Jf=nj.zeros([m,num_params]); // Jacobian of Func()  
    var input=nj.zeros([1,n]); // single row input  
  
    var last_mse = 0;  
  
    for(var i=0; i < MAX_ITER; i++) {  
        var mse = 0;  
  
        for(var j=0; j < m; j++) {  
            for(var k=0; k < n; k++) {  
                input.set(0,k,inputs.get(j,k));  
            }  
  
            r.set(j,0,outputs.get(j,0)-Func(input, params));  
  
            mse += r.get(j,0)*r.get(j,0);  
  
            for(var k=0; k < num_params; k++) {  
                Jf.set(j,k,Deriv(Func, input, params, k));  
            }  
        }  
  
        mse /= m;  
  
        // The difference in mse is very small, so quit  
        if(Math.abs(mse - last_mse) < 1e-8) {  
            break;  
        }  
  
//        var delta = getReMatrix(Jf.T.dot(Jf)).dot(Jf.T.dot(r)); 
        var Mat_Jf=Jf.flatten().tolist().toMatrix(Jf.shape[0],Jf.shape[1]);
        var Mat_r=r.flatten().tolist().toMatrix(r.shape[0]*r.shape[1],1);
        var Mat_delta = Mat_Jf.transpose().multiply(Mat_Jf).inverse().multiply(Mat_Jf.transpose()).multiply(Mat_r);
        delta=nj.array(Mat_delta.toArray()).reshape(params.shape[0],params.shape[1]);

        params=params.add(delta);  
  
        //printf("%d: mse=%f\n", i, mse);  
        console.log(mse);  
  
        last_mse = mse;  
    }  
    return params;
}  
