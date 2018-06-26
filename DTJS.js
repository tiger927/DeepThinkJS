//CNN SECTION START
function CNN(inp,out,data,label,hid,lr,actfun) {
	this.inp = inp;
	this.out = out;
	this.data = data;
	this.label = label;
	this.layers = [];
	this.train_count = 0;

	this.hid = hid;
	this.lr = lr;
	this.actfun = actfun;
	this.status = false;
}

CNN.prototype.train = function() {
	if(status) {
		
	}else{
		this.start();
		this.train();
	}
}

CNN.prototype.start = function() {
	this.status = true;
	this.forward();
	var inputs = this.fulConv();
	this.fulCon = new MLPNN(inputs.length,this.hid,this.out,inputs,this.labels,this.lr,this.actfun);
}

CNN.prototype.fulConV = function() {
	var datas = this.now.slice(0);
	var total = [];
	for(var i = 0;i < datas.length;i++) {
		var tmp = [];
		for(var j = 0;j < datas[i].length;j++) {
			for(var a = 0;a < datas[i][j].length;a++) {
				for(var b = 0;b < datas[i][j][a].length;b++) {
					tmp.push(datas[i][j][a][b]);
				}
			}
		}
		total.push(tmp);
	}
	return total;
}

CNN.prototype.addLayer = function(obj) {
	if(obj["name"] == "fConv") {
		var ker = obj["kernel"];
		var n = obj["number"];
		var stride = obj["stride"];
		var tmp = [];
		for(var i = 0;i < n;i++) {
			tmp.push(this.ranM(ker));
		}
		var biastmp = [];
		for(var i = 0;i < n;i++) {
			biastmp.push(Math.random() * 2 - 1);
		}
		this.layers.push({"s":stride,"type":"normal","content":tmp,"bias":biastmp});
	}else if(obj["name"] == "RELU") {
		this.layers.push({"type":"RELU"});
	}else if(obj["name"] == "pool") {
		var type = obj["type"];
		var n = obj["n"];
		this.layers.push({"type":"pool","method":type,"s":n});
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

CNN.prototype.deleteL = function() {
	this.layers.pop();
}

CNN.prototype.forward = function() {
	this.now = this.data.slice(0);
	for(var i = 0;i < this.layers.length;i++) {
		if(this.layers[i]["type"] == "normal") {
			var finaltmp = [];
			for(var c = 0;c < this.now.length;c++) {
				for(var j = 0;j < this.layers[i]["content"].length;j++) {
					var tmp2 = [];
					for(var a = 0;a <= this.now[c][0].length - this.layers[i]["content"][j].length;a++) {
						tmp2[a] = [];
						for(var b = 0;b <= this.now[c][0][a].length - this.layers[i]["content"][j].length;b++) {
							tmp2[a][b] = this.layers[i]["bias"][j];
						}
					}
					for(var k = 0;k < this.now[c].length;k++) {
						tmp2 = this.addA(this.applyF(this.now[c][k],this.layers[i]["content"][j],1),tmp2).slice(0);
					}
				}
				finaltmp.push(tmp2);
			}
			this.now = finaltmp.slice(0);
		}else if(this.layers[i]["type"] == "RELU") {
			for(var j = 0;j < this.now.length;j++) {
				for(var c = 0;c < this.now[j].length;c++) {
					for(var a = 0;a < this.now[j][c].length;a++) {
						for(var b = 0;b < this.now[j][c][a].length;b++) {
							this.now[j][c][a][b] = Math.max(this.now[j][c][a][b],0);
						}
					}
				}
			}
		}else if(this.layers[i]["type"] == "pool") {
			var s = this.layers[i]["s"];
			var method = this.layers[i]["method"];
			var tmp = [];
			for(var j = 0;j < this.now.length;j++) {
				var tmp2 = [];
				for(var e = 0;e < this.now[j].length;e++) {
					var tmp3 = [];
					for(var a = 0;a < this.now[j][e].length;a += s) {
						tmp3[a/s] = [];
						for(var b = 0;b < this.now[j][e][a].length;b += s) {
							var max = - Number.MAX_VALUE;
							var sum = 0;
							for(var c = 0;c < s&&a + c < this.now[j][e].length;c++) {
								for(var d = 0;d < s&&b + d < this.now[j][e][a].length;d++) {
									if(this.layers[i]["method"] == "max") {
										if(this.now[j][e][a + c][b + d] > max) {
											max = this.now[j][e][a + c][b + d];
										}
									}else if(this.layers[i]["method"] == "average") {
										sum += this.now[j][e][a + c][b + d];
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
					tmp2.push(tmp3);
				}
				tmp.push(tmp2);
			}
			this.now = tmp.slice(0);
		}
	}
}

CNN.prototype.applyF = function(graph,filter,s) {
	var result = [];
	for(var i = 0;i <= graph.length - filter.length;i += s) {
		result[i] = [];
		for(var j = 0;j <= graph.length - filter.length;j += s) {
			var tmp = 0;
			for(var a = 0;a < filter.length;a++) {
				for(var b = 0;b < filter.length;b++) {
					tmp += graph[a + i][b + j] * filter[a][b];
				}
			}
			result[i][j] = tmp / filter.length / filter.length;
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
	for(var k = 0;k < this.data.length;k++) {
		var result = [];
		for(var i = 0;i < this.data[k].length + 2 * x;i++) {
			result[i] = [];
			for(var j = 0;j < this.data[k][0].length + 2 * x;j++) {
				if(i >= x && i < this.data[k].length + x && j >= x && j < this.data[k][0].length + x) {
					result[i][j] = this.data[k][i - x][j - x];
				}else{
					result[i][j] = 0;
				}
			}
		}
		this.data[k] = result;
	}

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
	if(actfun == "sigmoid") {
		this.actfun = sigmoid;
	}else if(actfun == "tanh") {
		this.actfun = tanh;
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
	for(var i = this.nums - 1;i >= 1;i--) {
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

		for(var j = 0;j < this.nlayers[i];j++) {
			this.layers[i][j]["delta"] = errors[j] * this.actfun(this.layers[i][j]["output"],true);
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

//MLPNN SECTION END
