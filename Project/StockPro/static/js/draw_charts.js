
var companies = {"600718":"东软集团","000651":"格力电器","600839":"四川长虹","600320":"振华重工","601988":"中国银行",
                 "000066": "中国长城","601766":"中国中车","601390":"中国中铁","000768":"中航飞机","000063":"中兴通讯"};

function draw_chart(){
    //初始化echarts实例
    var myChart = echarts.init(document.getElementById("hist_futu"));

    // 指定图表的配置项和数据
     var option = {
        title: {
                text: companies[stock_code]+"("+stock_code+")" + "过去20天历史数据以及未来10天预测数据",
                textStyle:{
        　　　　  fontSize:15
                }
            },
        tooltip : {
                    trigger: 'item'
                },
        legend: {
                x : 'center',
                data:['过去20天','未来10天']
            },
        //工具框，可以选择
        toolbox: {
            feature: {
                saveAsImage: {}
            }
        },
        xAxis: {
            axisLabel: {
                rotate: 30,
                interval: 0
            },
            type: 'category',
            //boundaryGap: false,
            data: [] // x轴名称
        },
        yAxis: {
                type: 'value',
                axisLabel : {
                    formatter: '{value} 元'
                },
            }
        ,
        series: [
            {
            name:'过去20天',
            type: 'line',
            color:['#FF0000'],
            data: [],   // x坐标对应y值
            itemStyle : { normal: {label : {show: true}}},
            label: {
                    normal: {
                        show: true,
                        position: 'top'
                    }
                },
            },
            {
            name:'未来10天',
            data: [],   // x坐标对应y值
            itemStyle : { normal: {label : {show: true}}},
            type: 'line',
            label: {
                    normal: {
                        show: true,
                        position: 'top'
                    }
                },
            color:['#0000FF'],
            },
        ]
    };

    var min,max;
    for(var k=0; k <= 1; k++){
        if(k == 0){
            m_data = recent_data;
        }else{
            m_data = predict_data;
        }
        for(var i = 0 ; i < m_data.length; i++){
            var one_day = m_data[i];
            option['xAxis']['data'].push(one_day[0])
            if(k==0){
                option['series'][1]['data'].push(null);
            }
            option['series'][k]['data'].push(one_day[1].toFixed(2)) // toFixed(2)：保留两位小数（四舍五入）

            if(i == 0 && k == 0){
                min = max = one_day[1];
            }else{
                if(one_day[1] < min){
                    min = one_day[1];
                }
                if(one_day[1] > max){
                    max = one_day[1];
                }
            }
        }
    }

    option['yAxis']['min'] = parseInt(min);
    option['yAxis']['max'] = parseInt(max)+1;

    myChart.setOption(option);
}

if(recent_data != null && predict_data != null){
    draw_chart();
}

var ops = document.getElementById(stock_code);
ops.selected = true;


