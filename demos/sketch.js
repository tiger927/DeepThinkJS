var nn;
var datas = [];
var labels = [];

function setup() {
	for(var i = 0;i < 150;i++) {
    	var ran = i / 20;
    	datas.push([ran]);
    	//ran * ran / 5 - ran - 2 * Math.abs(Math.sin(ran * 2))
		labels.push([(ran * ran / 5 - ran - 2 * Math.abs(Math.sin(ran * 2)) + 1) * 0.3]);
	}
	createCanvas(1000,600);
	nn = new MLPNN(1,[10,9],1,datas,labels,0.03,"tanh");
	nn.init(true);
}

function draw() {
	background(51);
	for(var i = 0;i < 150;i++) {
		fill(255,255,255);
		noStroke();
		ellipse(i * 5 + 10,300 - (labels[i] * 200 + 10),5,5);
		noStroke();
		fill(255,0,0);
		ellipse(i * 5 + 10,300 - (nn.forward([i / 20])[0] * 200 + 10),4,4);
	}
	for(var i = 0;i < 20;i++) nn.train();
	stroke(255);
	line(0,90,1000,90);
	line(0,490,1000,490)
}
