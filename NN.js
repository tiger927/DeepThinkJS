var nodes = [];
var w = [];
var layers = [];
var num;
var lr = 0.03;

function NN(inp,hid,out,data,label) {
	this.inp = inp;
	this.hid = hid;
	this.out = out;
	this.data = data;
	this.label = label;
	num = 1 + hid.length + 1;
	for(var i = 0;i < num;i++) {
		if(i != 0 && i != num - 1) {
			layers[i] = this.hid[i - 1];
		}else if(i == 0){
			layers[i] = this.inp;
		}else{
			layers[i] = this.out;
		}
	}

	for(var i = 0;i < hid.length + 1;i++) {
		w[i] = [];
		for(var j = 0;j < layers[i];j++) {
			w[i][j] = [];
		}
	}
}

NN.prototype.init = function(rand) {
	for(var i = 0;i < w.length;i++) {
		for(var j = 0;j < layers[i];j++) {
			if(rand) {
				for(var k = 0;k < layers[i + 1];k++) {
					w[i][j][k] = Math.random();
				}
			}else{
				for(var k = 0;k < layers[i + 1];k++) {
					w[i][j][k] = 1;
				}
			}
		}
	}

	for(var i = 0;i < w.length + 1;i++) {
		nodes[i] = [];
		for(var j = 0;j < layers[i];j++) {
			nodes[i][j] = new node(i,j);
		}
	}
}

NN.prototype.train = function() {
	for(var i = 0;i < this.data.length;i++) {
		for(var j = 0;j < this.data[i].length;j++) {
			nodes[0][j].sum = this.data[i][j];
			nodes[0][j].activate();
		}
		for(var j = 0;j < this.label[i].length;j++) {
			var error = this.label[i][j] - nodes[num - 1][j].sum;
			nodes[num - 1][j].error = error;
			nodes[num - 1][j].activateE();
		}
		for(var j = 1;j < w.length + 1;j++) {
			for(var k = 0;k < layers[j];k++) {
				nodes[j][k].adjust();
			}
		}
		clearNodes();
	}
}

NN.prototype.result = function(data) {
	for(var i = 0;i < data.length;i++) {
		nodes[0][i].sum = data[i];
	}
	for(var i = 0;i < data.length;i++) {
		nodes[0][i].activate();
	}
	var res = [];
	for(var i = 0;i < this.out;i++) res[i] = nodes[this.hid.length + 1][i].sum;

	return res;
}



NN.prototype.cost = function() {
	var error = 0;
	for(var i = 0;i < this.data.length;i++) {
		var outs = this.result(this.data[i]);
		for(var j = 0;j < this.label[i].length;j++) {
			error += (this.label[i][j] - outs[j]) * (this.label[i][j] - outs[j]) / 2;
		}
	}
	return error;
}

function node(i,j) {
	this.i = i;
	this.j = j;
	this.count = 0;
	this.sum = 0;
	this.error = 0;
}

node.prototype.activate = function() {
	for(var k = 0;k < layers[this.i + 1];k++) {
		nodes[this.i + 1][k].sum += sigmoid(this.sum,false) * w[this.i][this.j][k];
		if(this.i + 1 != num - 1) {
			nodes[this.i + 1][k].count++;
			if(nodes[this.i + 1][k].count == layers[this.i]) {
				nodes[this.i + 1][k].activate();
			}
		}
	}
}

node.prototype.activateE = function() {
	for(var k = 0;k < layers[this.i - 1];k++) {
		nodes[this.i - 1][k].error += this.error * w[this.i - 1][k][this.j];
		if(this.i != 1) {
			nodes[this.i - 1][k].countE++;
			if(nodes[this.i - 1][k].countE == layers[this.i]) {
				nodes[this.i - 1][k].activateE();
			}
		}
	}
}


node.prototype.clear = function() {
	this.sum = 0;
	this.count = 0;
	this.error = 0;
	this.countE = 0;
}

node.prototype.adjust = function() {
	for(var i = 0;i < layers[this.i - 1];i++) {
		w[this.i - 1][i][this.j] += lr * sigmoid(nodes[this.i - 1][i].sum,true) * this.error * nodes[this.i - 1][i].sum;
	}
}

function clearNodes() {
	for(var i = 0;i < w.length + 1;i++) {
		for(var j = 0;j < layers[i];j++) {
			nodes[i][j].clear();
		}
	}
}

function ReLU(n,b) {
	if(b) {
		return Math.max(0,n) / n;
	}
	return Math.max(0,n);
}

function sigmoid(n,b) {
	if(b) {
		var tmp = sigmoid(n);
		return tmp * (1 - tmp);
	}
	return 1/(1 + Math.exp(-n));
}

function LReLU(n,b) {
	if(b) {
		if(n > 0) {
			return 1;
		}else{
			return 0.01;
		}
	}
	if(n > 0) {
		return n;
	}else{
		return 0.01 * n;
	}
}

