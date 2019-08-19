# GaussNewton.js


GaussNewton base on [numjs](https://github.com/nicolaspanel/numjs) and [matrixjs](https://github.com/Airblader/matrixjs)

## Sample

### test.html

```js
// For this demo we're going to try and fit to the function  
// F = A*exp(t*B)  
// There are 2 parameters: A B  
var num_params = 2;  

// Generate random data using these parameters  
var total_data = 8;  

var inputs=nj.zeros([total_data,1]);
var outputs=nj.zeros([total_data,1]);  

//load observation data  
for(var i=0; i < total_data; i++) {  
    inputs.set(i,0,i+1);  //load year  
}  
//load America population  
outputs.set(0,0,8.3);  
outputs.set(1,0,11.0);  
outputs.set(2,0,14.7);  
outputs.set(3,0,19.7);  
outputs.set(4,0,26.7);  
outputs.set(5,0,35.2);  
outputs.set(6,0,44.4);  
outputs.set(7,0,55.9);  

// Guess the parameters, it should be close to the true value, else it can fail for very sensitive functions!  
var params=nj.zeros([num_params,1]);

//init guess  
params.set(0,0,6);  
params.set(1,0,0.3);  

params=GaussNewton(Func, inputs, outputs, params);  

console.log("Parameters from GaussNewton:"+params.get(0,0)+", "
    +params.get(1,0));  

function Func(input, params) {  
    // Assumes input is a single row matrix  
    // Assumes params is a column matrix  
  
    var A = params.get(0,0);  
    var B = params.get(1,0);  
  
    var x = input.get(0,0);  
  
    return A*Math.exp(x*B);  
}  

> 15.913662615695564
> 0.7712413677491117
> 0.751641813304339
> 0.7516351496690794
> Parameters from GaussNewton:7.000153882089704, 0.26207659754628115
```
