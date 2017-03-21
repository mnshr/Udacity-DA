var parseDate = d3.time.format("%d-%b-%y").parse;
var svg1 = dimple.newSvg("#chart1", "100%", 530);

var grp_loans_by_credit_chart_1 = new dimple.chart(svg1);
grp_loans_by_credit_chart_1.setBounds(60, 35, "70%", 350);
	var grp_loans_by_credit_x_chart1 = grp_loans_by_credit_chart_1.addCategoryAxis("x", "Time");
	grp_loans_by_credit_x_chart1.title = "Quarterly Timeline";
	var grp_loans_by_credit_y_chart1 =grp_loans_by_credit_chart_1.addMeasureAxis("y", "num");
	grp_loans_by_credit_y_chart1 = "Loan Counts";
	var series1 = grp_loans_by_credit_chart_1.addSeries("Credit", dimple.plot.line);
   //series1.lineWeight=3;
   series1.lineMarkers=true;
   grp_loans_by_credit_chart_1.addLegend(60, 10, 500, 20, "right");
                                     
d3.csv("grp_loans_by_credit.csv", function(dataset) {
dataset.forEach(function(d) {
d.date = parseDate(d.Time);
});
   grp_loans_by_credit_chart_1.data = dataset;
   grp_loans_by_credit_chart_1.draw();
});    

var svg2 = dimple.newSvg("#chart2", "100%", 530);
var grp_loans_by_cat_chart_1 = new dimple.chart(svg2);
grp_loans_by_cat_chart_1.setBounds(60, 35, "70%", 350);
	var grp_loans_by_cat_x_chart1 = grp_loans_by_cat_chart_1.addCategoryAxis("x", "Time");
	grp_loans_by_cat_x_chart1.title = "Quarterly Timeline";
	var grp_loans_by_cat_y_chart1 =grp_loans_by_cat_chart_1.addMeasureAxis("y", "num");
	grp_loans_by_cat_y_chart1 = "Loan Counts";
	var series2 = grp_loans_by_cat_chart_1.addSeries("Category", dimple.plot.line);
   series2.lineMarkers=true;
   grp_loans_by_cat_chart_1.addLegend(60, 10, 500, 20, "right")
d3.csv("grp_loans_by_cat.csv", function(dataset) {
dataset.forEach(function(d) {
d.date = parseDate(d.Time);
});
   grp_loans_by_cat_chart_1.data = dataset;
   grp_loans_by_cat_chart_1.draw();
});    
    

function AverageLoan_Cr() {
  d3.select('svg').remove();
  //d3.selectAll('charts').remove();
  var svg1 = dimple.newSvg("#chart1", "100%", 530);
  var grp_loans_by_credit_chart_1 = new dimple.chart(svg1);
  grp_loans_by_credit_chart_1.setBounds(60, 35, "70%", 350);
	var grp_loans_by_credit_x_chart1 = grp_loans_by_credit_chart_1.addCategoryAxis("x", "Time");
	grp_loans_by_credit_x_chart1.title = "Quarterly Timeline";
	var grp_loans_by_credit_y_chart1 =grp_loans_by_credit_chart_1.addMeasureAxis("y", "num");
	grp_loans_by_credit_y_chart1 = "Loan Counts";
	var series1 = grp_loans_by_credit_chart_1.addSeries("Credit", dimple.plot.line);
   series1.lineMarkers=true;
    grp_loans_by_credit_chart_1.addLegend(60, 10, 500, 20, "right"); 
     d3.csv("grp_loans_by_credit.csv", function(dataset) {
        dataset.forEach(function(d) {
        d.date = parseDate(d.Time);
        });
           grp_loans_by_credit_chart_1.data = dataset;
           grp_loans_by_credit_chart_1.draw();
    });    

}
function AverageAmt_Cr() {
    d3.select('svg').remove();
    //d3.selectAll('charts').remove();
    var svg1 = dimple.newSvg("#chart1", "100%", 530);
    
 	var grp_prin_by_credit_chart_1 = new dimple.chart(svg1);
	grp_prin_by_credit_chart_1.setBounds(60, 35, "70%", 350);
	var grp_prin_by_credit_x_chart1 = grp_prin_by_credit_chart_1.addCategoryAxis("x", "Time");
	grp_prin_by_credit_x_chart1.title = "Quarterly Timeline";
	var grp_prin_by_credit_y_chart1=grp_prin_by_credit_chart_1.addMeasureAxis("y", "amt");
	grp_prin_by_credit_y_chart1= "Loan Original Amount";
	var series1 = grp_prin_by_credit_chart_1.addSeries("Credit", dimple.plot.line);
   series1.lineMarkers=true;
   grp_prin_by_credit_chart_1.addLegend(60, 10, 500, 20, "right"); 
   d3.csv("grp_prin_by_credit.csv", function(dataset) {
        dataset.forEach(function(d) {
        d.date = parseDate(d.Time);
        });
       grp_prin_by_credit_chart_1.data=dataset;
    	grp_prin_by_credit_chart_1.draw();
    
    });
}
function AverageDTI_Cr() {
    d3.select('svg').remove();

    var svg1 = dimple.newSvg("#chart1", "100%", 530);

	var chart_1 = new dimple.chart(svg1);
	chart_1.setBounds(60, 35, "70%", 350);
	var x_chart1 = chart_1.addCategoryAxis("x", "Time");
	x_chart1.title = "Quarterly Timeline";
	var y_chart1 =chart_1.addMeasureAxis("y", "dti");
	y_chart1 = "Debt to Income Ratio";
	var series1 = chart_1.addSeries("Credit", dimple.plot.line);
   series1.lineMarkers=true;
   chart_1.addLegend(60, 10, 500, 20, "right"); 
   d3.csv("grp_dti_by_credit.csv", function(dataset) {
        dataset.forEach(function(d) {
        d.date = parseDate(d.Time);
        });
       chart_1.data=dataset;
    	chart_1.draw();
    });

}
function AverageIncome_Cr() {
    d3.select('svg').remove();
    //d3.selectAll('charts').remove();
    var svg1 = dimple.newSvg("#chart1", "100%", 530);
 	 var chart_1 = new dimple.chart(svg1);
	 chart_1.setBounds(60, 35, "70%", 350);
    var x = chart_1.addCategoryAxis("x", "Time") 
	 x.title = "Quarterly Timeline";
	 var y_chart1 =chart_1.addMeasureAxis("y", "AvgIncome");
	 y_chart1.title = "Average Stated Income";
	 var series1 = chart_1.addSeries("Credit", dimple.plot.line);
    series1.lineMarkers=true;
    chart_1.addLegend(60, 10, 500, 20, "right"); 
    d3.csv("grp_inc_by_credit.csv", function(dataset) {
        dataset.forEach(function(d) {
        d.date = parseDate(d.Time);
        });
        chart_1.data=dataset;
        chart_1.draw();
    });
}
    
///////////////////////////////////////////////////////////////////////////////
function AverageLoan_Cat() {
    d3.selectAll('svg')[0][1].remove();
    d3.select('charts').remove();

    var svg2 = dimple.newSvg("#chart2", "100%", 530);
    var grp_loans_by_cat_chart_1 = new dimple.chart(svg2);
    grp_loans_by_cat_chart_1.setBounds(60, 35, "70%", 350);
    	var grp_loans_by_cat_x_chart1 = grp_loans_by_cat_chart_1.addCategoryAxis("x", "Time");
    	grp_loans_by_cat_x_chart1.title = "Quarterly Timeline2";
    	var grp_loans_by_cat_y_chart1 =grp_loans_by_cat_chart_1.addMeasureAxis("y", "num");
    	grp_loans_by_cat_y_chart1 = "Loan Counts";
    var series1 = grp_loans_by_cat_chart_1.addSeries("Category", dimple.plot.line);
    series1.lineMarkers=true;
    grp_loans_by_cat_chart_1.addLegend(60, 10, 500, 20, "right");                                          
    d3.csv("grp_loans_by_cat.csv", function(dataset) {
    dataset.forEach(function(d) {
    d.date = parseDate(d.Time);
    });
       grp_loans_by_cat_chart_1.data = dataset;
       grp_loans_by_cat_chart_1.draw();
    });    
}
function AverageAmt_Cat() {
    d3.selectAll('svg')[0][1].remove();
    d3.select('charts').remove();

    var svg2 = dimple.newSvg("#chart2", "100%", 530);
    var chart_1 = new dimple.chart(svg2);
    chart_1.setBounds(60, 35, "70%", 350);
    	var x_chart1 = chart_1.addCategoryAxis("x", "Time");
    	x_chart1.title = "Quarterly Timeline";
    	var y_chart1 =chart_1.addMeasureAxis("y", "prin");
    	y_chart1 = "Loan Original Amount";
    var series1 = chart_1.addSeries("Category", dimple.plot.line);
    series1.lineMarkers=true;
    chart_1.addLegend(60, 10, 500, 20, "right");     
    d3.csv("grp_prin_by_cat.csv", function(dataset) {
    dataset.forEach(function(d) {
    d.date = parseDate(d.Time);
    });
//debugger;
       chart_1.data = dataset;
       chart_1.draw();
    });
}
    
function AverageDTI_Cat() {
    d3.selectAll('svg')[0][1].remove();
    d3.select('charts').remove();
    var svg2 = dimple.newSvg("#chart2", "100%", 530);
    var chart_1 = new dimple.chart(svg2);
	
    chart_1.setBounds(60, 35, "70%", 350);
    var x_chart1 = chart_1.addCategoryAxis("x", "Time");
    x_chart1.title = "Quarterly Timeline";
    var y_chart1 =chart_1.addMeasureAxis("y", "dti");
    y_chart1 = "Debt to Income Ratio";
    var series1 = chart_1.addSeries("Category", dimple.plot.line);
    series1.lineMarkers=true;
    chart_1.addLegend(60, 10, 500, 20, "right"); 
    d3.csv("grp_dti_by_cat.csv", function(dataset) {
        dataset.forEach(function(d) {
        d.date = parseDate(d.Time);
        });
       chart_1.data=dataset;
    	chart_1.draw();
    });
}
function AverageIncome_Cat() {
       d3.select('charts').remove();
       d3.selectAll('svg')[0][1].remove();
    

    var svg2 = dimple.newSvg("#chart2", "100%", 530);
    var chart_1 = new dimple.chart(svg2);
	
    chart_1.setBounds(60, 35, "70%", 350);
    var x_chart1 = chart_1.addCategoryAxis("x", "Time");
    x_chart1.title = "Quarterly Timeline";
    var y_chart1 =chart_1.addMeasureAxis("y", "AvgIncome");
    y_chart1 = "Average Stated Income";
    var series1 = chart_1.addSeries("Category", dimple.plot.line);
    series1.lineMarkers=true;
    chart_1.addLegend(60, 10, 500, 20, "right"); 
    d3.csv("grp_inc_by_cat.csv", function(dataset) {
        dataset.forEach(function(d) {
        d.date = parseDate(d.Time);
        });
       chart_1.data=dataset;
    	chart_1.draw();
    });
}