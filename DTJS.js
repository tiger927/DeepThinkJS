//CNN SECTION START
function CNN(inp,out,data,label,hid,lr) {
	this.inp = inp;
	this.out = out;
	this.data = data;
	this.label = label;
	this.layers = [];
	this.train_count = 0;

	this.hid = hid;
	this.lr = lr;
	this.status = false;
}

CNN.prototype.addLayer = function(obj) {
	if(obj["name"] == "fConv") {
		var ker = obj["kernel"];
		var n = obj["number"];
		var stride = obj["stride"];
		var tmp = [];
		var nin = this.biggestO();
		for(var i = 0;i < n;i++) {
			tmp[i] = [];
			for(var j = 0;j < nin;j++) {
				tmp[i].push(this.ranM(ker));
			}
		}
		var biastmp = [];
		for(var i = 0;i < n;i++) {
			biastmp.push(Math.random() * 2 - 1);
		}
		this.layers.push({"s":stride,"type":"normal","content":tmp,"bias":biastmp,"n":n});
	}else if(obj["name"] == "RELU") {
		this.layers.push({"type":"RELU"});
	}else if(obj["name"] == "pool") {
		var type = obj["type"];
		var n = obj["n"];
		this.layers.push({"type":"pool","method":type,"s":n});
	}else if(obj["name"] == "padding") {
		var x = obj["x"];
		this.layers.push({"type":"padding","x":x});
	}else{
		console.log("Cannot recognize type");
	}
}

CNN.prototype.addA = function(arr1,arr2) {
	var result = [];
	for(var i = 0;i < arr1.length;i++) {
		result[i] = [];
		for(var j = 0;j < arr1[i].length;j++) {
			result[i][j] = arr1[i][j] + arr2[i][j];
		}
	}
	return result;
}

CNN.prototype.flip = function(matrix) {
	var result = []
	for(var i = matrix.length - 1;i >= 0;i--) {
		result[matrix.length - 1 - i] = [];
		for(var j = matrix[0].length - 1;j >= 0;j--) {
			result[matrix.length - 1 - i].push(matrix[i][j]);
		}
	}
	return result;
}

CNN.prototype.deleteL = function() {
	this.layers.pop();
}

CNN.prototype.forward = function() {
	this.now = this.data[0].slice(0);
	for(var i = 0;i < this.layers.length;i++) {
		this.runSL(i);
	}
}

CNN.prototype.runSL = function(i) {
	if(this.layers[i]["type"] == "normal") {

		var finaltmp = [];
		var n = this.layers[i]["n"];
		var filters = this.layers[i]["content"];
		var bias = this.layers[i]["bias"];
		var s = this.layers[i]["s"];
		for(var i = 0;i < n;i++) {
			var tmp = [];
			var nm = this.now[0].length - filters[i][0].length + 1;
			for(var j = 0;j < nm;j += s) {
				tmp[j / s] = [];
				for(var k = 0;k < nm;k += s) {
					tmp[j / s][k / s] = bias[i];
				}
			}
			for(var j = 0;j < this.now.length;j++) {
				tmp = this.addA(this.applyF(this.now[j],filters[i][j],s),tmp).slice(0);
			}
			finaltmp.push(tmp);
		}

		this.now = finaltmp.slice(0);
	}else if(this.layers[i]["type"] == "RELU") {
		for(var j = 0;j < this.now.length;j++) {
			for(var c = 0;c < this.now[j].length;c++) {
				for(var a = 0;a < this.now[j][c].length;a++) {
					this.now[j][c][a] = Math.max(this.now[j][c][a],0);
				}
			}
		}
	}else if(this.layers[i]["type"] == "pool") {
		var s = this.layers[i]["s"];
		var method = this.layers[i]["method"];
		var tmp = [];
		for(var j = 0;j < this.now.length;j++) {
			var tmp3 = [];
			for(var a = 0;a < this.now[j].length;a += s) {
				tmp3[a/s] = [];
				for(var b = 0;b < this.now[j][a].length;b += s) {
					var max = - Number.MAX_VALUE;
					var sum = 0;
					for(var c = 0;c < s&&a + c < this.now[j].length;c++) {
						for(var d = 0;d < s&&b + d < this.now[j][a].length;d++) {
							if(this.layers[i]["method"] == "max") {
								if(this.now[j][a + c][b + d] > max) {
									max = this.now[j][a + c][b + d];
								}
							}else if(this.layers[i]["method"] == "average") {
								sum += this.now[j][a + c][b + d];
							}
						}
					}
					if(this.layers[i]["method"] == "max") {
						tmp3[a/s][b/s] = max;
					}else if(this.layers[i]["method"] == "average") {
						tmp3[a/s][b/s] = sum / s / s;
					}
				}
			}
			tmp.push(tmp3);
		}
		this.now = tmp.slice(0);
	}else if(this.layers[i]["type"] == "padding") {
		var x = this.layers[i]["x"];
		var result = [];
		for(var k = 0;k < this.now.length;k++) {
			result[k] = [];
			for(var l = 0;l < this.now[k].length + 2 * x;l++) {
				result[k][l] = [];
				for(var j = 0;j < this.now[k][0].length + 2 * x;j++) {
					if(l >= x && l < this.now[k].length + x && j >= x && j < this.now[k][0].length + x) {
						result[k][l][j] = this.now[k][l - x][j - x];
					}else{
						result[k][l][j] = 0;
					}
				}
			}
		}
		this.now = result.slice(0);
	}
}

CNN.prototype.biggestO = function() {
	var nin = this.data[0].length;
	for(var i = this.layers.length - 1;i >= 0;i--) {
		if(this.layers[i]["type"] == "normal") {
			nin = this.layers[i]["n"];
			break;
		}
	}
	return nin;
}

CNN.prototype.applyF = function(graph,filter,s) {
	var result = [];
	for(var i = 0;i <= graph.length - filter.length;i += s) {
		result[i / s] = [];
		for(var j = 0;j <= graph.length - filter.length;j += s) {
			var tmp = 0;
			for(var a = 0;a < filter.length;a++) {
				for(var b = 0;b < filter.length;b++) {
					tmp += graph[a + i][b + j] * filter[a][b];
				}
			}
			result[i / s][j / s] = tmp;
		}
	}
	return result;

}

CNN.prototype.ranM = function(n) {
	var result = [];
	for(var i = 0;i < n;i++) {
		result[i] = [];
		for(var j = 0;j < n;j++) {
			result[i][j] = Math.random() * 2 - 1;
		}
	}
	return result;
}

CNN.prototype.padding = function(x) {
	

}
//CNN SECTION END



//KNN SECTION START

function KNN(data,label) {
	this.data = data;
	this.label = label;
}

KNN.prototype.normalize = function() {
	var small = [];
	var big = [];
	for(var i = 0;i < this.data[0].length;i++) {
		small[i] = Number.MAX_VALUE;
		big[i] = Number.MIN_VALUE;
	}
	for(var i = 0;i < this.data.length;i++) {	
		for(var j = 0;j < this.data[i].length;j++) {
			if(this.data[i][j] < small[j]) {
				small[j] = this.data[i][j];
			}
			if(this.data[i][j] > big[j]) {
				big[j] = this.data[i][j];
			}
		}
	}
	for(var i = 0;i < this.data[0].length;i++) {
		var dist = big[i] - small[i];
		for(var j = 0;j < this.data.length;j++) {
			this.data[j][i] = (this.data[j][i] - small[i]) / dist;
		}
	}
}

KNN.prototype.result = function(k,inp) {
	if(k > this.data.length) {
		console.log("Error: K is bigger than the number of datas");
		return;
	}
	var dist = [];
	for(var i = 0;i < this.data.length;i++) {
		dist[i] = this.dist(inp,this.data[i]);
	}

	var result = new Array(k);
	for(var i = 0;i < k;i++) {
		var ans;
		var min = Number.MAX_VALUE;
		for(var j = 0;j < dist.length;j++) {
			if(dist[j] < min) {
				min = dist[j];
				ans = j;
			}else if(dist[j] == min) {
				if(Math.random() >= 0.5) {
					min = dist[j];
					ans = j;
				}
			}
		}
		dist[ans] = Number.MAX_VALUE;
		result[i] = ans;
	}
	var count = new Object();
	var labels = [];
	for(var i = 0;i < result.length;i++) {
		if(count[this.label[result[i]]] == null) {
			count[this.label[result[i]]] = 0;
			labels.push(this.label[result[i]]);
		}
		count[this.label[result[i]]] = 0;
	}
	var ans = [];
	var numb = 0;
	for(var i = 0;i < labels.length;i++) {
		if(count[labels[i]] > numb) {
			ans = [];
			numb = count[labels[i]];
			ans.push(labels[i]);
		}else if(count[labels[i]] == numb) {
			ans.push(labels[i]);
		}
	}
	var final = Math.floor(Math.random() * ans.length);
	return ans[final];
}

KNN.prototype.dist = function(x,y) {
	var total = 0;
	for(var i = 0;i < x.length;i++) {
		total += (y[i] - x[i]) * (y[i] - x[i]);
	}
	return Math.sqrt(total);
}

//KNN SECTION END




//MLPNN SECTION START

function MLPNN(inp,hid,out,data,label,lr,actfun) {
	this.inp = inp;
	this.hid = hid;
	this.out = out;
	this.data = data;
	this.label = label;
	this.nums = 1 + hid.length + 1;
	this.lr = lr;
	this.layers = [];
	this.nlayers = [];
	this.train_count = 0;
	this.isCnn = false;
	if(actfun == "sigmoid") {
		this.actfun = sigmoid;
	}else if(actfun == "tanh") {
		this.actfun = tanh;
	}else if(actfun == "RELU") {
		this.actfun = RELU;
	}else{
		console.log("Activation Function Type cannot be identified");
	}
}

MLPNN.prototype.init = function(b) {
	for(var i = 0;i < this.nums;i++) {
		if(i == 0) {
			this.layers[i] = [];
			this.nlayers[i] = this.inp;
		}else if(i == this.nums - 1) {
			this.layers[i] = [];
			this.nlayers[i] = this.out;
		}else{
			this.layers[i] = [];
			this.nlayers[i] = this.hid[i - 1];
		}
	}
	for(var i = 1;i < this.nums;i++) {
		for(var j = 0;j < this.nlayers[i];j++) {
			var tmp = new Object();
			tmp["weights"] = [];
			for(var k = 0;k < this.nlayers[i - 1] + 1;k++) {
				if(b) {
					tmp["weights"].push(Math.random() * 2 - 1);
				}else{
					tmp["weights"].push(1);
				}
			}
			this.layers[i][j] = tmp;
		}
	}
}

MLPNN.prototype.activate = function(w,inp) {
	var result = w[0];
	for(var i = 1;i < w.length;i++) {
		result += w[i] * inp[i - 1];
	}
	return result;
}

MLPNN.prototype.backward = function(label) {
	for(var i = this.nums - 1;i >= 0;i--) {
		var errors = [];
		if(i == this.nums - 1) {
			for(var j = 0;j < this.out;j++) {
				var cur = this.layers[this.nums - 1][j];
				errors.push(label[j] - cur['output']);
			}
		}else{
			for(var j = 0;j < this.nlayers[i];j++) {
				var tmp = 0;
				for(var k = 0;k < this.nlayers[i + 1];k++) {
					var neuron = this.layers[i + 1][k];
					tmp += neuron["weights"][j + 1] * neuron["delta"];
				}
				errors.push(tmp);
			}
		}
		if(i != 0) {
			for(var j = 0;j < this.nlayers[i];j++) {
				this.layers[i][j]["delta"] = errors[j] * this.actfun(this.layers[i][j]["output"],true);
			}
		}

		if(i == 0 && this.isCnn) {
			this.cnnErr = errors;
		}

	}

}

MLPNN.prototype.clear = function(data) {
	for(var i = 1;i < this.nums;i++) {
		for(var j = 0;j < this.nlayers[i];j++) {
			delete this.layers[i][j].delta;
			delete this.layers[i][j].output;
		}
	}
}

MLPNN.prototype.forward = function(data) {
	var inputs = data;
	for(var i = 1;i < this.nums;i++) {
		var ninp = [];
		for(var j = 0;j < this.layers[i].length;j++) {
			var neuron = this.layers[i][j];
			var activation = this.activate(neuron["weights"],inputs);
			neuron["output"] = this.actfun(activation,false);
			ninp.push(neuron["output"]);
		}
		inputs = ninp;
	}
	return inputs;
}

MLPNN.prototype.train = function(o) {
	var error = 0;
	for(var j = 0;j < this.data.length;j++) {
		var cur = this.data[j];
		var output = this.forward(cur);
		for(var i = 0;i < this.label[j].length;i++) {
			error += ((this.label[j][i] - output[i]) * (this.label[j][i] - output[i]) / 2);
		}
		this.backward(this.label[j]);
		this.update(cur);
	}
	this.train_count++;
	this.cost = error;
	if(o) {
		console.log("Epoch " + this.train_count + " Cost: " + error);
	}
}

MLPNN.prototype.update = function(input) {
	for(var i = 1;i < this.nums;i++) {
		inputs = input.slice(0);
		if(i != 1) {
			inputs = [];
			for(var j = 0;j < this.nlayers[i - 1];j++) {
				var cur = this.layers[i - 1][j];
				inputs.push(cur["output"]);
			}
		}
		for(var j = 0;j < this.nlayers[i];j++) {
			for(var k = 1;k < this.layers[i][j]["weights"].length;k++) {
				this.layers[i][j]["weights"][k] += this.lr * this.layers[i][j]["delta"] * inputs[k - 1];
			}
			this.layers[i][j]["weights"][0] += this.lr * this.layers[i][j]["delta"];
		}
	}
}

function sigmoid(n,b) {
	if(b) {
		return n * (1 - n);
	}else{
		return 1/(1 + Math.exp(-n));
	}
}

function tanh(n,b) {
	if(b) {
		return 1 - n * n;
	}else{
		return Math.tanh(n);
	}
}

function RELU(n,b) {
	if(b) {
		if(n > 0) {
			return 1;
		}else{
			return 0;
		}
	}
	return Math.max(n,0);
}
//MLPNN SECTION END
