var parseDate = d3.time.format("%d-%b-%y").parse;
                              
var svg1 = dimple.newSvg("#chart1", "100%", 530);
var grp_loans_by_credit_chart_1 = new dimple.chart(svg1);
grp_loans_by_credit_chart_1.setBounds(60, 35, "70%", 350);
	var grp_loans_by_credit_x_chart1 = grp_loans_by_credit_chart_1.addCategoryAxis("x", "Time");
	grp_loans_by_credit_x_chart1.title = "Quarterly Timeline";
	var grp_loans_by_credit_y_chart1 =grp_loans_by_credit_chart_1.addMeasureAxis("y", "num");
	grp_loans_by_credit_y_chart1 = "Loan Counts";
	grp_loans_by_credit_chart_1.addSeries("Credit", dimple.plot.line);
                                     
d3.csv("grp_loans_by_credit.csv", function(dataset) {
    dataset.forEach(function(d) {
    d.date = parseDate(d.Time);
    });
   grp_loans_by_credit_chart_1.data = dataset;
   grp_loans_by_credit_chart_1.draw();
});    

////////////////////////////////////////////////////////////////
var svg2 = dimple.newSvg("#chart2", "100%", 530);
 	var grp_prin_by_credit_chart_1 = new dimple.chart(svg2);
	grp_prin_by_credit_chart_1.setBounds(60, 35, "70%", 350);
	var grp_prin_by_credit_x_chart1 = grp_prin_by_credit_chart_1.addCategoryAxis("x", "Time");
	grp_prin_by_credit_x_chart1.title = "Quarterly Timeline";
	var grp_prin_by_credit_y_chart1=grp_prin_by_credit_chart_1.addMeasureAxis("y", "amt");
	grp_prin_by_credit_y_chart1= "Loan Original Amount";
	grp_prin_by_credit_chart_1.addSeries("Credit", dimple.plot.line);
 
   d3.csv("grp_prin_by_credit.csv", function(dataset) {
    dataset.forEach(function(d) {
    d.date = parseDate(d.Time);
    });
   grp_prin_by_credit_chart_1.data=dataset;
	grp_prin_by_credit_chart_1.draw();

});
////////////////////////////////////////////////////////////////
var svg3 = dimple.newSvg("#chart3", "100%", 530);
	var grp_dti_by_credit_chart_1 = new dimple.chart(svg3);
	grp_dti_by_credit_chart_1.setBounds(60, 35, "70%", 350);
	
   var grp_dti_by_credit_x_chart_1 = grp_dti_by_credit_chart_1.addCategoryAxis("x", "Time");
	grp_dti_by_credit_x_chart_1.title = "Loan Status";
	var grp_dti_by_credit_y_chart_1=grp_dti_by_credit_chart_1.addMeasureAxis("y", "dti");
	grp_dti_by_credit_y_chart_1= "Debt to Income Ratio";
	
   grp_dti_by_credit_chart_1.addSeries("Credit", dimple.plot.line);

   d3.csv("grp_dti_by_credit.csv", function(dataset) {
    dataset.forEach(function(d) {
    d.date = parseDate(d.Time);
    });
   grp_dti_by_credit_chart_1.data=dataset;
	grp_dti_by_credit_chart_1.draw();
});
////////////////////////////////////////////////////////////////
var svg4 = dimple.newSvg("#chart4", "100%", 530);
 	 var grp_inc_by_credit_chart_1 = new dimple.chart(svg4);
	 grp_inc_by_credit_chart_1.setBounds(60, 35, "70%", 350);
                                     
var grp_inc_by_credit_x_chart_1 = grp_inc_by_credit_chart_1.addCategoryAxis("x", "Time") 
grp_inc_by_credit_x_chart_1.title = "Quarterly Timeline";
var grp_inc_by_credit_y_chart_1=grp_inc_by_credit_chart_1.addMeasureAxis("y", "AvgIncome");
grp_inc_by_credit_y_chart_1.title = "Average Stated Income";
     
grp_inc_by_credit_chart_1.addSeries("Credit", dimple.plot.line);

d3.csv("grp_inc_by_credit.csv", function(dataset) {
    dataset.forEach(function(d) {
    d.date = parseDate(d.Time);
    });
    grp_inc_by_credit_chart_1.data=dataset;
    grp_inc_by_credit_chart_1.draw();
});
    
///////////////////////////////////////////////////////////////////////////////
var svg5 = dimple.newSvg("#chart5", "100%", 530);
var grp_loans_by_cat_chart_1 = new dimple.chart(svg5);
grp_loans_by_cat_chart_1.setBounds(60, 35, "70%", 350);
	var grp_loans_by_cat_x_chart1 = grp_loans_by_cat_chart_1.addCategoryAxis("x", "Time");
	grp_loans_by_cat_x_chart1.title = "Quarterly Timeline";
	var grp_loans_by_cat_y_chart1 =grp_loans_by_cat_chart_1.addMeasureAxis("y", "num");
	grp_loans_by_cat_y_chart1 = "Loan Counts";
	
   grp_loans_by_cat_chart_1.addSeries("Category", dimple.plot.line);
                                     
d3.csv("grp_loans_by_cat.csv", function(dataset) {
    dataset.forEach(function(d) {
    d.date = parseDate(d.Time);
    });
   grp_loans_by_cat_chart_1.data = dataset;
   grp_loans_by_cat_chart_1.draw();
});    

///////////////////////////////////////////////////////////////////////////////
var svg6 = dimple.newSvg("#chart6", "100%", 530);
var grp_prin_by_cat_chart_1 = new dimple.chart(svg6);
grp_prin_by_cat_chart_1.setBounds(60, 35, "70%", 350);
	var grp_prin_by_cat_x_chart1 = grp_prin_by_cat_chart_1.addCategoryAxis("x", "Time");
	grp_prin_by_cat_x_chart1.title = "Quarterly Timeline2";
	var grp_prin_by_cat_y_chart1 =grp_prin_by_cat_chart_1.addMeasureAxis("y", "prin");
	grp_prin_by_cat_y_chart1 = "Loan Original Amount";
    
	grp_prin_by_cat_chart_1.addSeries("Category", dimple.plot.line);

d3.csv("grp_prin_by_cat.csv", function(dataset) {
    dataset.forEach(function(d) {
    d.date = parseDate(d.Time);
    });
    //debugger;
   grp_prin_by_cat_chart_1.data = dataset;
   grp_prin_by_cat_chart_1.draw();
});

///////////////////////////////////////////////////////////////////////////////
var svg7 = dimple.newSvg("#chart7", "100%", 530);
var grp_dti_by_cat_chart_1 = new dimple.chart(svg7);
	
grp_dti_by_cat_chart_1.setBounds(60, 35, "70%", 350);
var grp_dti_by_cat_x_chart1 = grp_dti_by_cat_chart_1.addCategoryAxis("x", "Time");
grp_dti_by_cat_x_chart1.title = "Quarterly Timeline";
var grp_dti_by_cat_y_chart1 =grp_dti_by_cat_chart_1.addMeasureAxis("y", "dti");
grp_dti_by_cat_y_chart1 = "Debt to Income Ratio";
grp_dti_by_cat_chart_1.addSeries("Category", dimple.plot.line);

d3.csv("grp_dti_by_cat.csv", function(dataset) {
    dataset.forEach(function(d) {
    d.date = parseDate(d.Time);
    });
   grp_dti_by_cat_chart_1.data=dataset;
	grp_dti_by_cat_chart_1.draw();
});
///////////////////////////////////////////////////////////////////////////////

var svg8 = dimple.newSvg("#chart8", "100%", 530);
var grp_inc_by_cat_chart_1 = new dimple.chart(svg2);
	
grp_inc_by_cat_chart_1.setBounds(60, 35, "70%", 350);
var grp_inc_by_cat_x_chart1 = grp_inc_by_cat_chart_1.addCategoryAxis("x", "Time");
grp_inc_by_cat_x_chart1.title = "Loan Status";
var grp_inc_by_cat_y_chart1 =grp_inc_by_cat_chart_1.addMeasureAxis("y", "AvgIncome");
grp_inc_by_cat_y_chart1 = "Average Stated Income";
grp_inc_by_cat_chart_1.addSeries("Category", dimple.plot.line);

d3.csv("grp_inc_by_cat.csv", function(dataset) {
dataset.forEach(function(d) {
d.date = parseDate(d.Time);
}); debugger;
   grp_inc_by_cat_chart_1.data=dataset;
	grp_inc_by_cat_chart_1.draw();
});