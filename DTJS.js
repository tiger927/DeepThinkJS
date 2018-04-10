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
	console.log("Hello world");

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
