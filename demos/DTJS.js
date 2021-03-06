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
