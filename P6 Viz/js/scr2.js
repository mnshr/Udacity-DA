var parseDate = d3.time.format("%d-%b-%y").parse;

//Create first chart as the page loads
var svg1 = dimple.newSvg("#chart1", "100%", 530);
var grp_loans_by_credit_chart_1 = new dimple.chart(svg1);
grp_loans_by_credit_chart_1.setBounds(60, 35, "70%", 350);
var grp_loans_by_credit_x_chart1 = grp_loans_by_credit_chart_1.addCategoryAxis("x", "Time");
grp_loans_by_credit_x_chart1.title = "Quarterly Timeline";
var grp_loans_by_credit_y_chart1 =grp_loans_by_credit_chart_1.addMeasureAxis("y", "num");
grp_loans_by_credit_y_chart1.title = "Loan Counts";
var series1 = grp_loans_by_credit_chart_1.addSeries("Credit", dimple.plot.line);
series1.lineMarkers=true;
grp_loans_by_credit_chart_1.addLegend(60, 10, 500, 20, "right");

//Load data from csv for default chart                                     
d3.csv("grp_loans_by_credit.csv", function(dataset) {
    dataset.forEach(function(d) {
        d.date = parseDate(d.Time);
    });
   grp_loans_by_credit_chart_1.data = dataset;
   grp_loans_by_credit_chart_1.draw();
});    

// Create header and description section for Number of Loans by Ratings
var chart1_div = document.getElementsByClassName('row_chart1')[0];
var html11 = '<div class="html11" id="html11"> \
    <H4>Number of Loans:</H4> Here is my text on number of loans</div>';
chart1_div.insertAdjacentHTML('afterbegin', html11);

//Create the 2nd default chart
var svg2 = dimple.newSvg("#chart2", "100%", 530);
var grp_loans_by_cat_chart_1 = new dimple.chart(svg2);

//This array is used to display customized tool tips for 2nd chart
var tooltip2 = ["Not Available", "Debt Consolidation", "Home Improvement",
                  "Business", "Personal Loan", "Student Use", "Auto", 
                  "Other", "Baby&Adoption", "Boat", "Cosmetic Procedure",
                   "Engagement Ring", "Green Loans", "Household Expenses", 
                   "Large Purchases", "Medical/Dental", "Motorcycle", "RV",
                   "Taxes", "Vacation", "Wedding Loans"
                   ];

grp_loans_by_cat_chart_1.setBounds(60, 35, "70%", 350);
var grp_loans_by_cat_x_chart1 = grp_loans_by_cat_chart_1.addCategoryAxis("x", "Time");
grp_loans_by_cat_x_chart1.title = "Quarterly Timeline";
var grp_loans_by_cat_y_chart1 =grp_loans_by_cat_chart_1.addMeasureAxis("y", "num");
grp_loans_by_cat_y_chart1.title = "Loan Counts";
var series2 = grp_loans_by_cat_chart_1.addSeries("Category", dimple.plot.line);
series2.lineMarkers=true;
series2.getTooltipText=function(e) {
           return [tooltip2[e.aggField[0]]];
};
grp_loans_by_cat_chart_1.addLegend(60, 10, 500, 20, "right")

//Coloring specific lines of interest and leaving others as grey
grp_loans_by_cat_chart_1.assignColor("1", "red")
grp_loans_by_cat_chart_1.assignColor("2", "blue")
grp_loans_by_cat_chart_1.assignColor("3", "green")
grp_loans_by_cat_chart_1.assignColor("4", "orange")
grp_loans_by_cat_chart_1.assignColor("7", "brown")
grp_loans_by_cat_chart_1.assignColor("13", "purple")
   
grp_loans_by_cat_chart_1.assignColor("0", "grey")
grp_loans_by_cat_chart_1.assignColor("5", "grey")
grp_loans_by_cat_chart_1.assignColor("6", "grey")
grp_loans_by_cat_chart_1.assignColor("8", "grey")
grp_loans_by_cat_chart_1.assignColor("9", "grey")
grp_loans_by_cat_chart_1.assignColor("10", "grey")
grp_loans_by_cat_chart_1.assignColor("11", "grey")
grp_loans_by_cat_chart_1.assignColor("12", "grey")
grp_loans_by_cat_chart_1.assignColor("14", "grey")
grp_loans_by_cat_chart_1.assignColor("15", "grey")
grp_loans_by_cat_chart_1.assignColor("16", "grey")
grp_loans_by_cat_chart_1.assignColor("17", "grey")
grp_loans_by_cat_chart_1.assignColor("18", "grey")
grp_loans_by_cat_chart_1.assignColor("19", "grey")
grp_loans_by_cat_chart_1.assignColor("20", "grey")

//Load the data from csv for 2nd default chart   
d3.csv("grp_loans_by_cat.csv", function(dataset) {
    dataset.forEach(function(d) {
        d.date = parseDate(d.Time);
    });
   grp_loans_by_cat_chart_1.data = dataset;
   grp_loans_by_cat_chart_1.draw();
});

// Create header and description section for Number of Loans by Loan Type
var chart1_div = document.getElementsByClassName('row_chart2')[0];
var html21 = '<div class="html21" id="html21"> \
    <H4>Number of Loans by Loan Type: </H4> Here is my text on number of loans</div>';
chart1_div.insertAdjacentHTML('afterbegin', html21);
    

//Function called when a menu for Average number of loans is clicked
function AverageLoan_Cr() {
      //Remove and recreate the chart
      d3.select('svg').remove();
      var svg1 = dimple.newSvg("#chart1", "100%", 530);

      var grp_loans_by_credit_chart_1 = new dimple.chart(svg1);
      grp_loans_by_credit_chart_1.setBounds(60, 35, "70%", 350);
      var grp_loans_by_credit_x_chart1 = grp_loans_by_credit_chart_1.addCategoryAxis("x", "Time");
      grp_loans_by_credit_x_chart1.title = "Quarterly Timeline";
      var grp_loans_by_credit_y_chart1 =grp_loans_by_credit_chart_1.addMeasureAxis("y", "num");
      grp_loans_by_credit_y_chart1.title = "Loan Counts";
      var series1 = grp_loans_by_credit_chart_1.addSeries("Credit", dimple.plot.line);
      series1.lineMarkers=true;
      grp_loans_by_credit_chart_1.addLegend(60, 10, 500, 20, "right"); 

      //Upload data from respective csv
      d3.csv("grp_loans_by_credit.csv", function(dataset) {
                dataset.forEach(function(d) {
                d.date = parseDate(d.Time);
          });
          grp_loans_by_credit_chart_1.data = dataset;
          grp_loans_by_credit_chart_1.draw();
      });    
      // Update header section
      h11 = document.getElementById('html11')
      h11.innerHTML="<H4>Number of Loans by Rating:</H4> Here is my text on number of loans";
}
                
//Function called when a menu for Average loan amount is clicked
function AverageAmt_Cr() {
   //Remove and recreate the chart
   d3.select('svg').remove();
   var svg1 = dimple.newSvg("#chart1", "100%", 530);

 	var grp_prin_by_credit_chart_1 = new dimple.chart(svg1);
	grp_prin_by_credit_chart_1.setBounds(60, 35, "70%", 350);
	var grp_prin_by_credit_x_chart1 = grp_prin_by_credit_chart_1.addCategoryAxis("x", "Time");
	grp_prin_by_credit_x_chart1.title = "Quarterly Timeline";
	var grp_prin_by_credit_y_chart1=grp_prin_by_credit_chart_1.addMeasureAxis("y", "amt");
	grp_prin_by_credit_y_chart1.title= "Loan Original Amount";
	var series1 = grp_prin_by_credit_chart_1.addSeries("Credit", dimple.plot.line);
   series1.lineMarkers=true;
   grp_prin_by_credit_chart_1.addLegend(60, 10, 500, 20, "right"); 

   //Upload data from respective csv
   d3.csv("grp_prin_by_credit.csv", function(dataset) {
       dataset.forEach(function(d) {
           d.date = parseDate(d.Time);
       });
       grp_prin_by_credit_chart_1.data=dataset;
    	 grp_prin_by_credit_chart_1.draw();
    
   });
   // Update header section
   h11 = document.getElementById('html11')
   h11.innerHTML="<H4>Average Amounts by Rating:</H4> Here is my text on loan amounts";
}
        
//Function called when a menu for Average Debt to Income is clicked
function AverageDTI_Cr() {
    //Remove and recreate the chart
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

    //Upload data from respective csv
    d3.csv("grp_dti_by_credit.csv", function(dataset) {
        dataset.forEach(function(d) {
            d.date = parseDate(d.Time);
        });
        chart_1.data=dataset;
        chart_1.draw();
    });
       
    // Update header section
    h11 = document.getElementById('html11')
    h11.innerHTML="<H4>Debt to Income by Rating:</H4> Here is my text on DTI";
}
        
//Function called when menu for Average income is clicked
function AverageIncome_Cr() {
    //Remove and recreate the chart
    d3.select('svg').remove();
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

    //Upload data from respective csv
    d3.csv("grp_inc_by_credit.csv", function(dataset) {
        dataset.forEach(function(d) {
            d.date = parseDate(d.Time);
        });
        chart_1.data=dataset;
        chart_1.draw();
    });
    // Update header section
    h11 = document.getElementById('html11')
    h11.innerHTML="<H4>Average Income by Rating:</H4> Here is my text on Average Income";
}