---
layout : post
title : "Nursing Analysis"
---
{% raw %}
    This notbook is to study the nursing occupation within Canada from 2014-23, based on data collected from various open datasets provided by the Government of Canada and affiliatde bodies. The aim of this analysis is to study the status and ongoing concerns regarding care provider burnout within Canadian Health infrastructure. 
    
    With the Government of Canada looking to increase investments towards Healthcare to revamp its digital infrastructure, it is important to find applicable use cases for all the data generated and capturde throught this intitiative. The final product is a dashboard application consisting of graphs highlighting key information that can have an impact on the study of burnout and workfore retention.

## EDA using various data sources


```python
import pandas as pd
import numpy as np
```


```python

```


```python
provinces=['N.L.','P.E.I.','N.S.','N.B.','Ont.','Man.','Sask.','Alta.','B.C.','Y.T.']
series_b=pd.read_excel('/content/hospital-spending-series-b-2005-2022-data-tables-en.xlsx',sheet_name=provinces,skiprows=4,nrows=19)
series_d=pd.read_excel('/content/hospital-spending-series-d-2009-2022-data-tables-en.xlsx',sheet_name=provinces,skiprows=3,nrows=15)
series_e=pd.read_excel('/content/hospital-spending-series-e-2009-2022-data-tables-en.xlsx',sheet_name=provinces,skiprows=4,nrows=15)
```


```python
from functools import reduce
series=[]
for sr in [series_b,series_d,series_e]:
  df=[]
  for k in sr.keys():
    sr[k]['Province']=str(k)
    df_k=pd.DataFrame(sr[k])
    df_k[df_k.columns] = df_k[df_k.columns].astype(str)
    df.append(df_k)
  sr=reduce(lambda left,right: pd.merge(left,right,on=df[0].columns.tolist(),how='outer'),df)
  series.append(sr)

```


```python
series_b=series[0]
series_d=series[1]
series_e=series[2]
```


```python
col=series_b.columns.tolist()[:15]
series_b=series_b[col]
series_b=series_b[series_b.Year != 'Annual percentage change by year']
series_e=series_e[series_e.Year != 'Annual percentage change by year']
```


```python
series_e=series_e.add_prefix('FTE_')
series_e.rename(columns={'FTE_Year':'Year','FTE_Province':'Province'},inplace=True)
```


```python

series_e
```





  <div id="df-93329f99-2a30-4f9a-b0cb-e4b5eb94503f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>FTE_Administrative Services</th>
      <th>FTE_Support Services</th>
      <th>FTE_Nursing Inpatient Services</th>
      <th>FTE_Intensive Care Nursing Unit</th>
      <th>FTE_Operating Room</th>
      <th>FTE_Long-Term Care Nursing Unit</th>
      <th>FTE_Emergency</th>
      <th>FTE_Other Ambulatory Care Services</th>
      <th>FTE_Medical Imaging</th>
      <th>FTE_Other Diagnostic and Therapeutic</th>
      <th>FTE_Community Health Services</th>
      <th>FTE_Research, Education \nand Other</th>
      <th>FTE_Total</th>
      <th>Province</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2009–2010</td>
      <td>1021.0</td>
      <td>4006.6</td>
      <td>4858.1</td>
      <td>856.8</td>
      <td>879.1</td>
      <td>324.8</td>
      <td>942.2</td>
      <td>1322.1</td>
      <td>708.7</td>
      <td>3017.1</td>
      <td>2080.0</td>
      <td>479.1</td>
      <td>20495.5</td>
      <td>N.S.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2009–2010</td>
      <td>1165.8</td>
      <td>5493.8</td>
      <td>5837.4</td>
      <td>885.7</td>
      <td>982.1</td>
      <td>297.6</td>
      <td>899.4</td>
      <td>1457.0</td>
      <td>593.0</td>
      <td>1839.2</td>
      <td>2134.2</td>
      <td>330.8</td>
      <td>21916.1</td>
      <td>Man.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009–2010</td>
      <td>161.2</td>
      <td>658.5</td>
      <td>786.7</td>
      <td>54.0</td>
      <td>98.5</td>
      <td>28.3</td>
      <td>105.7</td>
      <td>107.8</td>
      <td>118.4</td>
      <td>304.2</td>
      <td>175.3</td>
      <td>26.7</td>
      <td>2625.3</td>
      <td>P.E.I.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2009–2010</td>
      <td>1714.9</td>
      <td>11939.9</td>
      <td>14510.0</td>
      <td>2491.9</td>
      <td>2294.4</td>
      <td>2587.7</td>
      <td>2258.9</td>
      <td>3759.9</td>
      <td>1246.6</td>
      <td>8459.2</td>
      <td>842.4</td>
      <td>1301.7</td>
      <td>53407.6</td>
      <td>Alta.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2009–2010</td>
      <td>18.4</td>
      <td>77.8</td>
      <td>70.0</td>
      <td>0.0</td>
      <td>15.3</td>
      <td>0.0</td>
      <td>19.9</td>
      <td>8.7</td>
      <td>15.4</td>
      <td>41.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>266.7</td>
      <td>Y.T.</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>136</th>
      <td>2022–2023</td>
      <td>42.6</td>
      <td>124.9</td>
      <td>97.3</td>
      <td>11.8</td>
      <td>55.1</td>
      <td>0.0</td>
      <td>44.4</td>
      <td>18.0</td>
      <td>29.4</td>
      <td>71.5</td>
      <td>0.0</td>
      <td>4.1</td>
      <td>499.2</td>
      <td>Y.T.</td>
    </tr>
    <tr>
      <th>137</th>
      <td>2022–2023</td>
      <td>511.6</td>
      <td>3076.0</td>
      <td>2200.9</td>
      <td>493.6</td>
      <td>507.8</td>
      <td>745.4</td>
      <td>587.7</td>
      <td>677.8</td>
      <td>406.9</td>
      <td>1713.5</td>
      <td>180.0</td>
      <td>224.2</td>
      <td>11325.4</td>
      <td>N.L.</td>
    </tr>
    <tr>
      <th>138</th>
      <td>2022–2023</td>
      <td>745.1</td>
      <td>4378.1</td>
      <td>3457.0</td>
      <td>604.3</td>
      <td>628.3</td>
      <td>624.2</td>
      <td>812.0</td>
      <td>1287.0</td>
      <td>697.4</td>
      <td>2677.8</td>
      <td>751.3</td>
      <td>376.2</td>
      <td>17038.7</td>
      <td>N.B.</td>
    </tr>
    <tr>
      <th>139</th>
      <td>2022–2023</td>
      <td>988.2</td>
      <td>3952.2</td>
      <td>4902.9</td>
      <td>736.8</td>
      <td>898.5</td>
      <td>328.3</td>
      <td>1117.1</td>
      <td>1713.3</td>
      <td>647.6</td>
      <td>3349.7</td>
      <td>3595.8</td>
      <td>482.0</td>
      <td>22712.4</td>
      <td>N.S.</td>
    </tr>
    <tr>
      <th>149</th>
      <td>Note</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>nan</td>
      <td>B.C.</td>
    </tr>
  </tbody>
</table>
<p>141 rows × 15 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-93329f99-2a30-4f9a-b0cb-e4b5eb94503f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-93329f99-2a30-4f9a-b0cb-e4b5eb94503f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-93329f99-2a30-4f9a-b0cb-e4b5eb94503f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-1dc1b6cf-e761-4f95-913a-59456a95b3ee">
  <button class="colab-df-quickchart" onclick="quickchart('df-1dc1b6cf-e761-4f95-913a-59456a95b3ee')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-1dc1b6cf-e761-4f95-913a-59456a95b3ee button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_52dd119a-9c2a-4d0d-8ede-274d51f862b7">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('series_e')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_52dd119a-9c2a-4d0d-8ede-274d51f862b7 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('series_e');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
hospital_exp=pd.merge(series_b,series_e,on=['Year','Province'],how='outer')
hospital_exp.drop(hospital_exp.tail(1).index,inplace=True)
hospital_exp['Year']=hospital_exp['Year'].apply(lambda x: int(x.split('–')[0]))

```


```python
hospital_exp=hospital_exp[hospital_exp.Year >=2014]
hospital_exp
```





  <div id="df-7ed49fbb-c143-4939-a853-a7f2ca2e6509" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Administrative Services</th>
      <th>Support Services</th>
      <th>Nursing Inpatient Services</th>
      <th>Intensive Care Nursing Unit</th>
      <th>Operating Room</th>
      <th>Long-Term Care Nursing Unit</th>
      <th>Emergency</th>
      <th>Other Ambulatory Care Services</th>
      <th>Medical Imaging</th>
      <th>...</th>
      <th>FTE_Intensive Care Nursing Unit</th>
      <th>FTE_Operating Room</th>
      <th>FTE_Long-Term Care Nursing Unit</th>
      <th>FTE_Emergency</th>
      <th>FTE_Other Ambulatory Care Services</th>
      <th>FTE_Medical Imaging</th>
      <th>FTE_Other Diagnostic and Therapeutic</th>
      <th>FTE_Community Health Services</th>
      <th>FTE_Research, Education \nand Other</th>
      <th>FTE_Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>110</th>
      <td>2014</td>
      <td>306.6</td>
      <td>1672.7</td>
      <td>1825.3</td>
      <td>417.9</td>
      <td>549.2</td>
      <td>154.6</td>
      <td>363.0</td>
      <td>665.6</td>
      <td>340.8</td>
      <td>...</td>
      <td>3086.2</td>
      <td>2655.7</td>
      <td>1734.2</td>
      <td>2794.0</td>
      <td>4553.4</td>
      <td>1985.9</td>
      <td>8945.5</td>
      <td>498.9</td>
      <td>1345.8</td>
      <td>58881.7</td>
    </tr>
    <tr>
      <th>111</th>
      <td>2014</td>
      <td>262.6</td>
      <td>1204.1</td>
      <td>1490.1</td>
      <td>336.2</td>
      <td>558.1</td>
      <td>248.1</td>
      <td>327.7</td>
      <td>415.2</td>
      <td>300.1</td>
      <td>...</td>
      <td>2701.3</td>
      <td>2525.0</td>
      <td>3242.4</td>
      <td>2854.3</td>
      <td>3452.6</td>
      <td>2388.2</td>
      <td>8237.1</td>
      <td>934.4</td>
      <td>1168.1</td>
      <td>52678.0</td>
    </tr>
    <tr>
      <th>112</th>
      <td>2014</td>
      <td>125.5</td>
      <td>497.4</td>
      <td>547.8</td>
      <td>108.6</td>
      <td>170.3</td>
      <td>23.7</td>
      <td>105.1</td>
      <td>205.1</td>
      <td>78.6</td>
      <td>...</td>
      <td>1068.7</td>
      <td>996.4</td>
      <td>305.0</td>
      <td>1152.5</td>
      <td>1726.1</td>
      <td>595.1</td>
      <td>2066.0</td>
      <td>2339.6</td>
      <td>484.8</td>
      <td>24617.3</td>
    </tr>
    <tr>
      <th>113</th>
      <td>2014</td>
      <td>69.6</td>
      <td>305.7</td>
      <td>307.3</td>
      <td>58.6</td>
      <td>108.1</td>
      <td>26.3</td>
      <td>70.3</td>
      <td>148.5</td>
      <td>80.4</td>
      <td>...</td>
      <td>599.9</td>
      <td>581.9</td>
      <td>377.7</td>
      <td>769.9</td>
      <td>947.0</td>
      <td>690.9</td>
      <td>2480.3</td>
      <td>493.1</td>
      <td>363.0</td>
      <td>15756.6</td>
    </tr>
    <tr>
      <th>114</th>
      <td>2014</td>
      <td>59.3</td>
      <td>303.1</td>
      <td>241.0</td>
      <td>55.8</td>
      <td>87.1</td>
      <td>49.9</td>
      <td>54.5</td>
      <td>105.1</td>
      <td>49.7</td>
      <td>...</td>
      <td>515.2</td>
      <td>483.7</td>
      <td>644.2</td>
      <td>509.3</td>
      <td>777.6</td>
      <td>396.9</td>
      <td>1719.0</td>
      <td>86.9</td>
      <td>229.9</td>
      <td>11447.9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>2022</td>
      <td>134.5</td>
      <td>558.1</td>
      <td>560.9</td>
      <td>104.5</td>
      <td>207.3</td>
      <td>33.0</td>
      <td>149.4</td>
      <td>322.2</td>
      <td>91.7</td>
      <td>...</td>
      <td>736.8</td>
      <td>898.5</td>
      <td>328.3</td>
      <td>1117.1</td>
      <td>1713.3</td>
      <td>647.6</td>
      <td>3349.7</td>
      <td>3595.8</td>
      <td>482.0</td>
      <td>22712.4</td>
    </tr>
    <tr>
      <th>196</th>
      <td>2022</td>
      <td>1942.7</td>
      <td>5354.7</td>
      <td>5612.7</td>
      <td>1460.4</td>
      <td>2017.3</td>
      <td>721.2</td>
      <td>1394.0</td>
      <td>2884.6</td>
      <td>1592.6</td>
      <td>...</td>
      <td>9796.5</td>
      <td>7965.5</td>
      <td>5891.1</td>
      <td>10142.8</td>
      <td>14175.5</td>
      <td>8125.7</td>
      <td>27192.2</td>
      <td>12850.0</td>
      <td>8868.9</td>
      <td>198611.6</td>
    </tr>
    <tr>
      <th>197</th>
      <td>2022</td>
      <td>17.9</td>
      <td>94.2</td>
      <td>86.5</td>
      <td>8.2</td>
      <td>18.9</td>
      <td>n/r</td>
      <td>18.2</td>
      <td>40.7</td>
      <td>10.2</td>
      <td>...</td>
      <td>70.5</td>
      <td>97.5</td>
      <td>n/r</td>
      <td>150.5</td>
      <td>250.8</td>
      <td>100.5</td>
      <td>386.7</td>
      <td>177.3</td>
      <td>26.0</td>
      <td>2949.0</td>
    </tr>
    <tr>
      <th>198</th>
      <td>2022</td>
      <td>125.2</td>
      <td>504.4</td>
      <td>673.0</td>
      <td>137.8</td>
      <td>209.5</td>
      <td>16.8</td>
      <td>130.1</td>
      <td>144.6</td>
      <td>122.0</td>
      <td>...</td>
      <td>1038.9</td>
      <td>915.0</td>
      <td>37.2</td>
      <td>923.6</td>
      <td>1167.3</td>
      <td>776.9</td>
      <td>3305.7</td>
      <td>1510.6</td>
      <td>190.5</td>
      <td>25518.8</td>
    </tr>
    <tr>
      <th>199</th>
      <td>2022</td>
      <td>9.1</td>
      <td>22.7</td>
      <td>14.8</td>
      <td>2.3</td>
      <td>10.1</td>
      <td>n/r</td>
      <td>8.2</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>...</td>
      <td>11.8</td>
      <td>55.1</td>
      <td>0.0</td>
      <td>44.4</td>
      <td>18.0</td>
      <td>29.4</td>
      <td>71.5</td>
      <td>0.0</td>
      <td>4.1</td>
      <td>499.2</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 28 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-7ed49fbb-c143-4939-a853-a7f2ca2e6509')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-7ed49fbb-c143-4939-a853-a7f2ca2e6509 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-7ed49fbb-c143-4939-a853-a7f2ca2e6509');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e3a20594-19fa-4db2-948d-4b2e2501a751">
  <button class="colab-df-quickchart" onclick="quickchart('df-e3a20594-19fa-4db2-948d-4b2e2501a751')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e3a20594-19fa-4db2-948d-4b2e2501a751 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_37ccef4e-3be9-4b54-99a1-b3cb715513ad">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('hospital_exp')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_37ccef4e-3be9-4b54-99a1-b3cb715513ad button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('hospital_exp');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
series_d=series[1]
series_d=series_d[series_d['Year '] != 'Notes']
series_d.rename(columns={'Year ':'Year'},inplace=True)

```

    <ipython-input-40-3dce083cc05d>:3: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    


```python
hospital_exp.reset_index(inplace=True)
```


```python
hospital_exp.drop('index',axis=1,inplace=True)
```


```python
import re
columns=hospital_exp.columns.tolist()
columns.remove('Year')
columns.remove('Province')

for col in columns:
  if hospital_exp[col].dtype =='O':
    hospital_exp[col]=hospital_exp[col].apply(lambda x: float(re.findall("\d+\.\d+",str(x))[0])
                                             if re.findall("\d+\.\d+",str(x)) else np.nan)
hospital_exp
```





  <div id="df-63ce9a54-0463-452d-adea-28943dad3500" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Administrative Services</th>
      <th>Support Services</th>
      <th>Nursing Inpatient Services</th>
      <th>Intensive Care Nursing Unit</th>
      <th>Operating Room</th>
      <th>Long-Term Care Nursing Unit</th>
      <th>Emergency</th>
      <th>Other Ambulatory Care Services</th>
      <th>Medical Imaging</th>
      <th>...</th>
      <th>FTE_Intensive Care Nursing Unit</th>
      <th>FTE_Operating Room</th>
      <th>FTE_Long-Term Care Nursing Unit</th>
      <th>FTE_Emergency</th>
      <th>FTE_Other Ambulatory Care Services</th>
      <th>FTE_Medical Imaging</th>
      <th>FTE_Other Diagnostic and Therapeutic</th>
      <th>FTE_Community Health Services</th>
      <th>FTE_Research, Education \nand Other</th>
      <th>FTE_Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>306.6</td>
      <td>1672.7</td>
      <td>1825.3</td>
      <td>417.9</td>
      <td>549.2</td>
      <td>154.6</td>
      <td>363.0</td>
      <td>665.6</td>
      <td>340.8</td>
      <td>...</td>
      <td>3086.2</td>
      <td>2655.7</td>
      <td>1734.2</td>
      <td>2794.0</td>
      <td>4553.4</td>
      <td>1985.9</td>
      <td>8945.5</td>
      <td>498.9</td>
      <td>1345.8</td>
      <td>58881.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014</td>
      <td>262.6</td>
      <td>1204.1</td>
      <td>1490.1</td>
      <td>336.2</td>
      <td>558.1</td>
      <td>248.1</td>
      <td>327.7</td>
      <td>415.2</td>
      <td>300.1</td>
      <td>...</td>
      <td>2701.3</td>
      <td>2525.0</td>
      <td>3242.4</td>
      <td>2854.3</td>
      <td>3452.6</td>
      <td>2388.2</td>
      <td>8237.1</td>
      <td>934.4</td>
      <td>1168.1</td>
      <td>52678.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>125.5</td>
      <td>497.4</td>
      <td>547.8</td>
      <td>108.6</td>
      <td>170.3</td>
      <td>23.7</td>
      <td>105.1</td>
      <td>205.1</td>
      <td>78.6</td>
      <td>...</td>
      <td>1068.7</td>
      <td>996.4</td>
      <td>305.0</td>
      <td>1152.5</td>
      <td>1726.1</td>
      <td>595.1</td>
      <td>2066.0</td>
      <td>2339.6</td>
      <td>484.8</td>
      <td>24617.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014</td>
      <td>69.6</td>
      <td>305.7</td>
      <td>307.3</td>
      <td>58.6</td>
      <td>108.1</td>
      <td>26.3</td>
      <td>70.3</td>
      <td>148.5</td>
      <td>80.4</td>
      <td>...</td>
      <td>599.9</td>
      <td>581.9</td>
      <td>377.7</td>
      <td>769.9</td>
      <td>947.0</td>
      <td>690.9</td>
      <td>2480.3</td>
      <td>493.1</td>
      <td>363.0</td>
      <td>15756.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>59.3</td>
      <td>303.1</td>
      <td>241.0</td>
      <td>55.8</td>
      <td>87.1</td>
      <td>49.9</td>
      <td>54.5</td>
      <td>105.1</td>
      <td>49.7</td>
      <td>...</td>
      <td>515.2</td>
      <td>483.7</td>
      <td>644.2</td>
      <td>509.3</td>
      <td>777.6</td>
      <td>396.9</td>
      <td>1719.0</td>
      <td>86.9</td>
      <td>229.9</td>
      <td>11447.9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2022</td>
      <td>134.5</td>
      <td>558.1</td>
      <td>560.9</td>
      <td>104.5</td>
      <td>207.3</td>
      <td>33.0</td>
      <td>149.4</td>
      <td>322.2</td>
      <td>91.7</td>
      <td>...</td>
      <td>736.8</td>
      <td>898.5</td>
      <td>328.3</td>
      <td>1117.1</td>
      <td>1713.3</td>
      <td>647.6</td>
      <td>3349.7</td>
      <td>3595.8</td>
      <td>482.0</td>
      <td>22712.4</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2022</td>
      <td>1942.7</td>
      <td>5354.7</td>
      <td>5612.7</td>
      <td>1460.4</td>
      <td>2017.3</td>
      <td>721.2</td>
      <td>1394.0</td>
      <td>2884.6</td>
      <td>1592.6</td>
      <td>...</td>
      <td>9796.5</td>
      <td>7965.5</td>
      <td>5891.1</td>
      <td>10142.8</td>
      <td>14175.5</td>
      <td>8125.7</td>
      <td>27192.2</td>
      <td>12850.0</td>
      <td>8868.9</td>
      <td>198611.6</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2022</td>
      <td>17.9</td>
      <td>94.2</td>
      <td>86.5</td>
      <td>8.2</td>
      <td>18.9</td>
      <td>NaN</td>
      <td>18.2</td>
      <td>40.7</td>
      <td>10.2</td>
      <td>...</td>
      <td>70.5</td>
      <td>97.5</td>
      <td>NaN</td>
      <td>150.5</td>
      <td>250.8</td>
      <td>100.5</td>
      <td>386.7</td>
      <td>177.3</td>
      <td>26.0</td>
      <td>2949.0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2022</td>
      <td>125.2</td>
      <td>504.4</td>
      <td>673.0</td>
      <td>137.8</td>
      <td>209.5</td>
      <td>16.8</td>
      <td>130.1</td>
      <td>144.6</td>
      <td>122.0</td>
      <td>...</td>
      <td>1038.9</td>
      <td>915.0</td>
      <td>37.2</td>
      <td>923.6</td>
      <td>1167.3</td>
      <td>776.9</td>
      <td>3305.7</td>
      <td>1510.6</td>
      <td>190.5</td>
      <td>25518.8</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2022</td>
      <td>9.1</td>
      <td>22.7</td>
      <td>14.8</td>
      <td>2.3</td>
      <td>10.1</td>
      <td>NaN</td>
      <td>8.2</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>...</td>
      <td>11.8</td>
      <td>55.1</td>
      <td>0.0</td>
      <td>44.4</td>
      <td>18.0</td>
      <td>29.4</td>
      <td>71.5</td>
      <td>0.0</td>
      <td>4.1</td>
      <td>499.2</td>
    </tr>
  </tbody>
</table>
<p>90 rows × 28 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-63ce9a54-0463-452d-adea-28943dad3500')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-63ce9a54-0463-452d-adea-28943dad3500 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-63ce9a54-0463-452d-adea-28943dad3500');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-afb0c395-93d0-4409-9ded-f7ffa638f22b">
  <button class="colab-df-quickchart" onclick="quickchart('df-afb0c395-93d0-4409-9ded-f7ffa638f22b')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-afb0c395-93d0-4409-9ded-f7ffa638f22b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_23bc3f17-99bb-4ac5-aeaa-809fba55484d">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('hospital_exp')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_23bc3f17-99bb-4ac5-aeaa-809fba55484d button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('hospital_exp');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
b_col=series_b.columns.tolist()[:15]
b_col.remove('Year')
b_col.remove('Province')
for col in b_col:
  fte_col='FTE_'+col
  name=str(col)+'_comp'
  hospital_exp[name]=(hospital_exp[col]/hospital_exp[fte_col])*1e6
```


```python
hos_col=hospital_exp.columns.tolist()
nurse_col=[x for x in hos_col if 'Nursing' in x and 'comp' in x]
print(hos_col)
nurse_col
```

    ['Year', 'Administrative Services', 'Support Services', 'Nursing Inpatient Services', 'Intensive Care Nursing Unit', 'Operating Room', 'Long-Term Care Nursing Unit', 'Emergency', 'Other Ambulatory Care Services', 'Medical Imaging', 'Other Diagnostic and Therapeutic', 'Community Health Services', 'Research, Education \nand Other ', 'Total', 'Province', 'FTE_Administrative Services', 'FTE_Support Services', 'FTE_Nursing Inpatient Services', 'FTE_Intensive Care Nursing Unit', 'FTE_Operating Room', 'FTE_Long-Term Care Nursing Unit', 'FTE_Emergency', 'FTE_Other Ambulatory Care Services', 'FTE_Medical Imaging', 'FTE_Other Diagnostic and Therapeutic', 'FTE_Community Health Services', 'FTE_Research, Education \nand Other ', 'FTE_Total', 'Administrative Services_comp', 'Support Services_comp', 'Nursing Inpatient Services_comp', 'Intensive Care Nursing Unit_comp', 'Operating Room_comp', 'Long-Term Care Nursing Unit_comp', 'Emergency_comp', 'Other Ambulatory Care Services_comp', 'Medical Imaging_comp', 'Other Diagnostic and Therapeutic_comp', 'Community Health Services_comp', 'Research, Education \nand Other _comp', 'Total_comp']
    




    ['Nursing Inpatient Services_comp',
     'Intensive Care Nursing Unit_comp',
     'Long-Term Care Nursing Unit_comp']




```python
province_comp=hospital_exp.groupby(['Year'])['Total_comp'].mean().reset_index()
#province_comp.reset_index(inplace=True)
province_comp
```





  <div id="df-ac255b06-2580-4aab-bf9f-0d885604ea1e" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Total_comp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>122165.685451</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>123723.207031</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>125165.934039</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>128590.461705</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018</td>
      <td>129904.539743</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019</td>
      <td>133386.931989</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020</td>
      <td>136129.146040</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2021</td>
      <td>139989.168559</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2022</td>
      <td>146098.391250</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ac255b06-2580-4aab-bf9f-0d885604ea1e')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ac255b06-2580-4aab-bf9f-0d885604ea1e button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ac255b06-2580-4aab-bf9f-0d885604ea1e');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-60f63df2-e495-4076-a691-e227f41e4f9a">
  <button class="colab-df-quickchart" onclick="quickchart('df-60f63df2-e495-4076-a691-e227f41e4f9a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-60f63df2-e495-4076-a691-e227f41e4f9a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_5cffbdbb-838c-43bf-b764-7a26c2f66754">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('province_comp')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_5cffbdbb-838c-43bf-b764-7a26c2f66754 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('province_comp');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
province_comp.loc[province_comp.Total_comp.nlargest(20).index]
```





  <div id="df-8db07b48-3ad1-4b51-ad14-5148776e5014" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Total_comp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>2022</td>
      <td>146098.391250</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2021</td>
      <td>139989.168559</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020</td>
      <td>136129.146040</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019</td>
      <td>133386.931989</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018</td>
      <td>129904.539743</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>128590.461705</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>125165.934039</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>123723.207031</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>122165.685451</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-8db07b48-3ad1-4b51-ad14-5148776e5014')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-8db07b48-3ad1-4b51-ad14-5148776e5014 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-8db07b48-3ad1-4b51-ad14-5148776e5014');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-2004bbbe-5fd4-4bd9-878b-dc768c9507d3">
  <button class="colab-df-quickchart" onclick="quickchart('df-2004bbbe-5fd4-4bd9-878b-dc768c9507d3')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-2004bbbe-5fd4-4bd9-878b-dc768c9507d3 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
hospital_exp.Year.unique()
```




    array([2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])




```python
workforce_rn=pd.read_excel('/content/nursing-in-canada-2014-2023-data-tables-en.xlsx',sheet_name='5 Workforce',skiprows=1,nrows=460)

workforce_rn['Year']=workforce_rn['Year'].astype('int')
workforce_rn['Workforce: number \nof nurses']=pd.to_numeric(workforce_rn['Workforce: number \nof nurses'].astype(str).replace('-',np.nan),
                                                             errors='coerce')
```


```python
workforce_total=workforce_rn[workforce_rn.Jurisdiction== 'Provinces/territories with available data'][['Year','Jurisdiction','Type of professional','Workforce: number \nof nurses']]

workforce_total=workforce_total.groupby(['Year'])['Workforce: number \nof nurses'].sum().reset_index()
workforce_total['Year']=workforce_total['Year'].astype('int')
workforce_total
```





  <div id="df-0779230a-5f04-4eb7-861d-d99c88fe0598" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Workforce: number \nof nurses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>383846.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>390352.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>396178.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>398787.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018</td>
      <td>402855.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019</td>
      <td>396048.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020</td>
      <td>406094.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2021</td>
      <td>413411.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2022</td>
      <td>419745.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023</td>
      <td>429946.0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-0779230a-5f04-4eb7-861d-d99c88fe0598')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-0779230a-5f04-4eb7-861d-d99c88fe0598 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-0779230a-5f04-4eb7-861d-d99c88fe0598');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-1f5063f0-9a13-4709-864b-efe0f9535ecd">
  <button class="colab-df-quickchart" onclick="quickchart('df-1f5063f0-9a13-4709-864b-efe0f9535ecd')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-1f5063f0-9a13-4709-864b-efe0f9535ecd button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_98b5a722-6517-488c-ab38-b5947f01d828">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('workforce_total')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_98b5a722-6517-488c-ab38-b5947f01d828 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('workforce_total');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
workforce_total.Year.dtypes
```




    dtype('int64')




```python
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig=make_subplots(specs=[[{'secondary_y':True}]])

x=workforce_total.Year
y1=workforce_total['Workforce: number \nof nurses']
y2=province_comp.Total_comp

fig.add_trace(go.Scatter(x=x,y=y1,mode='lines+markers',name= 'Workforce'),secondary_y=False)
fig.add_trace(go.Bar(x=x,y=y2,opacity=0.4,name='Compensation'),secondary_y=True)

fig.update_yaxes(title_text='Total Number of Professionals',secondary_y=False)
fig.update_yaxes(title_text='Mean Compensation of Professionals',secondary_y=True)
fig.update_xaxes(title_text='Year')

fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="f46aab01-7def-4c87-ad86-0b4fc50d0b73" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("f46aab01-7def-4c87-ad86-0b4fc50d0b73")) {                    Plotly.newPlot(                        "f46aab01-7def-4c87-ad86-0b4fc50d0b73",                        [{"mode":"lines+markers","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[383846.0,390352.0,396178.0,398787.0,402855.0,396048.0,406094.0,413411.0,419745.0,429946.0],"type":"scatter","xaxis":"x","yaxis":"y"},{"opacity":0.4,"x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[122165.68545102554,123723.20703060064,125165.93403876985,128590.46170466398,129904.53974263878,133386.93198899267,136129.14604048486,139989.16855932522,146098.39125015153],"type":"bar","xaxis":"x","yaxis":"y2"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.94],"title":{"text":"Year"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Total Number of Professionals"}},"yaxis2":{"anchor":"x","overlaying":"y","side":"right","title":{"text":"Mean Compensation of Professionals"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('f46aab01-7def-4c87-ad86-0b4fc50d0b73');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


# Salary Data collection & cleaning


```python
!pip install ckanapi
```

    Collecting ckanapi
      Downloading ckanapi-4.8-py3-none-any.whl.metadata (618 bytes)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from ckanapi) (75.1.0)
    Collecting docopt (from ckanapi)
      Downloading docopt-0.6.2.tar.gz (25 kB)
      Preparing metadata (setup.py) ... [?25l[?25hdone
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from ckanapi) (2.32.3)
    Requirement already satisfied: six<2.0,>=1.9 in /usr/local/lib/python3.10/dist-packages (from ckanapi) (1.16.0)
    Collecting simplejson (from ckanapi)
      Downloading simplejson-3.19.3-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.2 kB)
    Requirement already satisfied: python-slugify>=1.0 in /usr/local/lib/python3.10/dist-packages (from ckanapi) (8.0.4)
    Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.10/dist-packages (from python-slugify>=1.0->ckanapi) (1.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->ckanapi) (3.4.0)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->ckanapi) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->ckanapi) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->ckanapi) (2024.8.30)
    Downloading ckanapi-4.8-py3-none-any.whl (46 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m46.3/46.3 kB[0m [31m2.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading simplejson-3.19.3-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (137 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m137.9/137.9 kB[0m [31m8.9 MB/s[0m eta [36m0:00:00[0m
    [?25hBuilding wheels for collected packages: docopt
      Building wheel for docopt (setup.py) ... [?25l[?25hdone
      Created wheel for docopt: filename=docopt-0.6.2-py2.py3-none-any.whl size=13706 sha256=8355d98eca513e7ab22f5dec7e0bc884834604b42b390976ce87ab502c4bd66b
      Stored in directory: /root/.cache/pip/wheels/fc/ab/d4/5da2067ac95b36618c629a5f93f809425700506f72c9732fac
    Successfully built docopt
    Installing collected packages: docopt, simplejson, ckanapi
    Successfully installed ckanapi-4.8 docopt-0.6.2 simplejson-3.19.3
    


```python
import requests
```


```python
!pip install ckanapi
import requests

url="https://open.canada.ca/data/en/dataset/adad580f-76b0-4502-bd05-20c125de9116"
rec_id="adad580f-76b0-4502-bd05-20c125de9116"
base_url = "https://open.canada.ca/data/api/3/action"

def get_metadata(rec_id):
  url=f"{base_url}/package_show"
  params={'id':rec_id}
  options={'Accept':'app/json','Accept-Language':'en'}
  response=requests.get(url,params=params,headers=options)
  response.raise_for_status()
  return response.json()

def get_data(data_id):
  url=f"{base_url}/datastore_search"
  params={"resource_id":data_id,"limit":100000}
  response=requests.get(url,params=params)
  response.raise_for_status()
  return response.json()


```


```python

```


```python
import numpy as np
import pandas as pd

meta=get_metadata(rec_id)
resources=meta['result']['resources']
salary_dict={}
keys=['salary_2014','salary_2015','salary_2016','salary_2017','salary_2018','salary_2019','salary_2020','salary_2021','salary_2022','salary_2023']
years=np.arange(2014,2024).tolist()
years=[str(x) for x in years]
for k in keys:
  for i in resources:
    if i['name'].split()[0] in years:
      data_id=i['id']
      data=get_data(data_id)
      records=data['result']['records']
      salary_dict[i['name']]=pd.DataFrame(records)

noc_codes=['NOC_32101','NOC_31301','NOC_31302','NOC_3012','NOC_3233','NOC_3151','NOC_3152']
```


```python
Occupation_dict={'Head Nurses and Supervisors': 'NOC_3151',
 'Licensed Practical Nurse': ['NOC_32101', 'NOC_3233'],
 'Nurse Practitioners': 'NOC_31302',
 'Registered Nurse and Registered Psych. Nurse': ['NOC_31301', 'NOC_3012', 'NOC_3152']}
```


```python
for name,df in salary_dict.items():
  df.columns=df.columns.str.upper()
  df.columns=df.columns.str.replace(' ','_')
  df.columns=df.columns.str.replace('-','_')
  if 'NOC_CNP_2006' in df.columns:
    salary_dict[name]=df[df['NOC_CNP_2006'].isin(noc_codes)]
  else:
    salary_dict[name]=df[df['NOC_CNP'].isin(noc_codes)]
  if 'NOC_TITLE_ENG' in df.columns:
    df.rename(columns={'NOC_Title_ENG': 'NOC_Title'},inplace=True)

  elif 'NOC Title' in df.columns :
    df.rename(columns={'NOC Title': 'NOC_Title'},inplace=True)

  elif 'NOC_CNP_2006' in df.columns:
    df.rename(columns={'NOC_CNP_2006': 'NOC_CNP'},inplace=True)

  elif 'NOC_Title_E' in df.columns:
    df.rename(columns={'NOC_TITLE_ENG': 'NOC_Title'},inplace=True)

  salary_dict[name]=df
#######################################################################################################################
for col in ['LOW_WAGE_SALAIRE_MINIUM','MEDIAN_WAGE_SALAIRE_MEDIAN', 'HIGH_WAGE_SALAIRE_MAXIMAL']:

  df[col]=pd.to_numeric(df[col],errors='coerce')
  if 'NOC_CNP_2006' in df.columns:
    new_df=df.groupby(['NO_CNP_2006','PROV'])[['LOW_WAGE_SALAIRE_MINIUM','MEDIAN_WAGE_SALAIRE_MEDIAN', 'HIGH_WAGE_SALAIRE_MAXIMAL']].mean().reset_index()
  else:
    new_df=df.groupby(['NOC_CNP','PROV'])[['LOW_WAGE_SALAIRE_MINIUM','MEDIAN_WAGE_SALAIRE_MEDIAN', 'HIGH_WAGE_SALAIRE_MAXIMAL']].mean().reset_index()

  salary_dict[name]=new_df
#######################################################################################################################
Occupation_dict={'Head Nurses and Supervisors': 'NOC_3151',
 'Licensed Practical Nurse': ['NOC_32101', 'NOC_3233'],
 'Nurse Practitioners': 'NOC_31302',
 'Registered Nurse and Registered Psych. Nurse': ['NOC_31301', 'NOC_3012', 'NOC_3152']}

for name,df in salary_dict.items():
  yr=int(name.split()[0])
  df['Year']=yr
  if 'NOC_CNP_2006' in df.columns:
    df.rename(columns={'NOC_CNP_2006': 'NOC_CNP'},inplace=True)

  df['Occupation']=df['NOC_CNP'].map(Occupation_dict)
  salary_dict[name]=df
```


```python


```

    <ipython-input-59-a7b6db62c88d>:6: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    


```python
for name,df in salary_dict.items():
  df.columns=df.columns.str.upper()
  df.columns=df.columns.str.replace(' ','_')
  df.columns=df.columns.str.replace('-','_')
  for col in ['LOW_WAGE_SALAIRE_MINIUM','MEDIAN_WAGE_SALAIRE_MEDIAN', 'HIGH_WAGE_SALAIRE_MAXIMAL']:

    df[col]=pd.to_numeric(df[col],errors='coerce')
  if 'NOC_CNP_2006' in df.columns:
    new_df=df.groupby(['NO_CNP_2006','PROV'])[['LOW_WAGE_SALAIRE_MINIUM','MEDIAN_WAGE_SALAIRE_MEDIAN', 'HIGH_WAGE_SALAIRE_MAXIMAL']].mean().reset_index()
  else:
    new_df=df.groupby(['NOC_CNP','PROV'])[['LOW_WAGE_SALAIRE_MINIUM','MEDIAN_WAGE_SALAIRE_MEDIAN', 'HIGH_WAGE_SALAIRE_MAXIMAL']].mean().reset_index()
  salary_dict[name]=new_df
```


```python
occ=can_salary.groupby(['Occupation'])['NOC_CNP'].unique().reset_index()
Occupation_dict={}
for i,row in occ.iterrows():
  Occupation_dict[row['Occupation']]=row['NOC_CNP']

Occupation_dict
```




    {'Head Nurses and Supervisors': array(['NOC_3151'], dtype=object),
     'Licensed Practical Nurse': array(['NOC_32101', 'NOC_3233'], dtype=object),
     'Nurse Practitioners': array(['NOC_31302'], dtype=object),
     'Registered Nurse and Registered Psych. Nurse': array(['NOC_31301', 'NOC_3012', 'NOC_3152'], dtype=object)}




```python
Occupation_dict={'Head Nurses and Supervisors': 'NOC_3151',
 'Licensed Practical Nurse': ['NOC_32101', 'NOC_3233'],
 'Nurse Practitioners': 'NOC_31302',
 'Registered Nurse and Registered Psych. Nurse': ['NOC_31301', 'NOC_3012', 'NOC_3152']}

for name,df in salary_dict.items():
  yr=int(name.split()[0])
  df['Year']=yr

  if 'NOC_CNP_2006' in df.columns:
    df.rename(columns={'NOC_CNP_2006': 'NOC_CNP'},inplace=True)

  df['Occupation']=df['NOC_CNP'].map(Occupation_dict)
  salary_dict[name]=df

```


```python
can_salary=pd.concat(salary_dict.values()).reset_index(drop=True)
can_salary
```





  <div id="df-676d87f6-368e-4a0f-991d-ef77e3622951" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NOC_CNP</th>
      <th>PROV</th>
      <th>LOW_WAGE_SALAIRE_MINIUM</th>
      <th>MEDIAN_WAGE_SALAIRE_MEDIAN</th>
      <th>HIGH_WAGE_SALAIRE_MAXIMAL</th>
      <th>Year</th>
      <th>Occupation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NOC_31301</td>
      <td>AB</td>
      <td>30.606667</td>
      <td>46.263333</td>
      <td>51.658889</td>
      <td>2023</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NOC_31301</td>
      <td>BC</td>
      <td>32.595556</td>
      <td>42.193333</td>
      <td>50.306667</td>
      <td>2023</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NOC_31301</td>
      <td>MB</td>
      <td>32.435556</td>
      <td>42.002222</td>
      <td>50.088889</td>
      <td>2023</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NOC_31301</td>
      <td>NA</td>
      <td>28.000000</td>
      <td>40.390000</td>
      <td>50.000000</td>
      <td>2023</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NOC_31301</td>
      <td>NB</td>
      <td>30.635000</td>
      <td>40.066667</td>
      <td>46.925000</td>
      <td>2023</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>317</th>
      <td>NOC_3233</td>
      <td>ON</td>
      <td>20.662583</td>
      <td>26.292833</td>
      <td>31.310325</td>
      <td>2014</td>
      <td>Licensed Practical Nurse</td>
    </tr>
    <tr>
      <th>318</th>
      <td>NOC_3233</td>
      <td>PE</td>
      <td>19.310000</td>
      <td>23.330000</td>
      <td>34.000000</td>
      <td>2014</td>
      <td>Licensed Practical Nurse</td>
    </tr>
    <tr>
      <th>319</th>
      <td>NOC_3233</td>
      <td>QC</td>
      <td>18.017857</td>
      <td>21.879286</td>
      <td>27.232857</td>
      <td>2014</td>
      <td>Licensed Practical Nurse</td>
    </tr>
    <tr>
      <th>320</th>
      <td>NOC_3233</td>
      <td>SK</td>
      <td>21.253333</td>
      <td>32.550000</td>
      <td>34.980000</td>
      <td>2014</td>
      <td>Licensed Practical Nurse</td>
    </tr>
    <tr>
      <th>321</th>
      <td>NOC_3233</td>
      <td>YT</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014</td>
      <td>Licensed Practical Nurse</td>
    </tr>
  </tbody>
</table>
<p>322 rows × 7 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-676d87f6-368e-4a0f-991d-ef77e3622951')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-676d87f6-368e-4a0f-991d-ef77e3622951 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-676d87f6-368e-4a0f-991d-ef77e3622951');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-b996003d-8fdf-4b29-b44f-106f7c522c82">
  <button class="colab-df-quickchart" onclick="quickchart('df-b996003d-8fdf-4b29-b44f-106f7c522c82')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-b996003d-8fdf-4b29-b44f-106f7c522c82 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_2cfe5868-53b9-4b47-8676-48ec6fe2058a">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('can_salary')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_2cfe5868-53b9-4b47-8676-48ec6fe2058a button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('can_salary');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
yearly_sal=can_salary.groupby(['Year'])[['MEDIAN_WAGE_SALAIRE_MEDIAN','HIGH_WAGE_SALAIRE_MAXIMAL','LOW_WAGE_SALAIRE_MINIUM']].mean().reset_index()
yearly_sal
```





  <div id="df-da06e458-6670-45fc-aeaf-25c0ee34d821" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>MEDIAN_WAGE_SALAIRE_MEDIAN</th>
      <th>HIGH_WAGE_SALAIRE_MAXIMAL</th>
      <th>LOW_WAGE_SALAIRE_MINIUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>34.442483</td>
      <td>42.699192</td>
      <td>21.067040</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>35.035598</td>
      <td>41.419449</td>
      <td>21.281010</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>34.013480</td>
      <td>40.186268</td>
      <td>23.686018</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>34.613198</td>
      <td>40.969562</td>
      <td>24.626508</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018</td>
      <td>35.741522</td>
      <td>42.036706</td>
      <td>24.941992</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019</td>
      <td>35.911955</td>
      <td>42.376967</td>
      <td>24.901906</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020</td>
      <td>36.361477</td>
      <td>42.712994</td>
      <td>25.833515</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2021</td>
      <td>36.927546</td>
      <td>42.862204</td>
      <td>27.625408</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2022</td>
      <td>36.833887</td>
      <td>41.861431</td>
      <td>28.022174</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023</td>
      <td>42.607722</td>
      <td>48.536538</td>
      <td>30.728031</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-da06e458-6670-45fc-aeaf-25c0ee34d821')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-da06e458-6670-45fc-aeaf-25c0ee34d821 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-da06e458-6670-45fc-aeaf-25c0ee34d821');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-a7fc5c1b-5c24-4dc3-9113-caaefc7219e1">
  <button class="colab-df-quickchart" onclick="quickchart('df-a7fc5c1b-5c24-4dc3-9113-caaefc7219e1')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-a7fc5c1b-5c24-4dc3-9113-caaefc7219e1 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_4b2b4ed5-065e-4fa5-a196-7dcad24dc7b9">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('yearly_sal')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_4b2b4ed5-065e-4fa5-a196-7dcad24dc7b9 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('yearly_sal');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig3=make_subplots(specs=[[{'secondary_y':True}]])

x=workforce_total.Year
y1=workforce_total['Workforce: number \nof nurses']
y2_1=yearly_sal.LOW_WAGE_SALAIRE_MINIUM*(37.75*52)
y2_3=yearly_sal.HIGH_WAGE_SALAIRE_MAXIMAL*(37.75*52)
y2_2=yearly_sal.MEDIAN_WAGE_SALAIRE_MEDIAN*(37.75*52)


fig3.add_trace(go.Scatter(x=x,y=y2_1,mode='markers'),secondary_y=True)
fig3.add_trace(go.Scatter(x=x,y=y2_2,mode='markers'),secondary_y=True)
fig3.add_trace(go.Scatter(x=x,y=y2_3,mode='markers'),secondary_y=True)
fig3.add_trace(go.Scatter(x=x,y=y1,mode='lines+markers'),secondary_y=False)

fig3.update_yaxes(title_text='Total Number of Professionals',secondary_y=False)
fig3.update_yaxes(title_text='Mean Compensation of Professionals',secondary_y=True)
fig3.update_xaxes(title_text='Year')

fig3.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="ecdd3edb-135a-49b7-9994-9050a274e9b7" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("ecdd3edb-135a-49b7-9994-9050a274e9b7")) {                    Plotly.newPlot(                        "ecdd3edb-135a-49b7-9994-9050a274e9b7",                        [{"mode":"markers","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[41354.59977507392,41774.62229111552,46495.654302413335,48341.83588665945,48961.13073222222,48882.440917142856,50711.18978920635,54228.67671412698,55007.52847449597,60319.12440128172],"type":"scatter","xaxis":"x","yaxis":"y2"},{"mode":"markers","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[67610.59343923125,68774.8787703373,66768.46072525778,67945.70703949496,70160.60812222223,70495.16688603176,71377.57900825398,72488.77223714285,72304.91974796834,83638.95749227236],"type":"scatter","xaxis":"x","yaxis":"y2"},{"mode":"markers","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[83818.51465322841,81306.37933967152,78885.64313054286,80423.24959415584,82518.05331714285,83185.98556666667,83845.60778285714,84138.50576650794,82173.98934306808,95277.22385606062],"type":"scatter","xaxis":"x","yaxis":"y2"},{"mode":"lines+markers","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[383846.0,390352.0,396178.0,398787.0,402855.0,396048.0,406094.0,413411.0,419745.0,429946.0],"type":"scatter","xaxis":"x","yaxis":"y"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.94],"title":{"text":"Year"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Total Number of Professionals"}},"yaxis2":{"anchor":"x","overlaying":"y","side":"right","title":{"text":"Mean Compensation of Professionals"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('ecdd3edb-135a-49b7-9994-9050a274e9b7');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```python
occ_salary=can_salary.groupby(['Year','Occupation'])[['MEDIAN_WAGE_SALAIRE_MEDIAN','HIGH_WAGE_SALAIRE_MAXIMAL','LOW_WAGE_SALAIRE_MINIUM']].mean().reset_index()
occ_salary
```





  <div id="df-31d42687-b994-49e0-95ab-193b5bc56eb0" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Occupation</th>
      <th>MEDIAN_WAGE_SALAIRE_MEDIAN</th>
      <th>HIGH_WAGE_SALAIRE_MAXIMAL</th>
      <th>LOW_WAGE_SALAIRE_MINIUM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>Head Nurses and Supervisors</td>
      <td>37.596182</td>
      <td>46.718868</td>
      <td>17.897773</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014</td>
      <td>Licensed Practical Nurse</td>
      <td>25.244920</td>
      <td>30.613562</td>
      <td>19.234585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>39.191233</td>
      <td>49.036728</td>
      <td>24.996965</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>Head Nurses and Supervisors</td>
      <td>38.574091</td>
      <td>45.763636</td>
      <td>18.475000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>Licensed Practical Nurse</td>
      <td>25.389028</td>
      <td>28.948096</td>
      <td>19.450590</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>39.834801</td>
      <td>47.805080</td>
      <td>24.923919</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016</td>
      <td>Licensed Practical Nurse</td>
      <td>25.967133</td>
      <td>29.789329</td>
      <td>20.813424</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2016</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>40.335609</td>
      <td>48.355291</td>
      <td>25.943057</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2017</td>
      <td>Licensed Practical Nurse</td>
      <td>26.933805</td>
      <td>31.022911</td>
      <td>21.422727</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2017</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>40.647006</td>
      <td>48.784787</td>
      <td>27.143765</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2018</td>
      <td>Licensed Practical Nurse</td>
      <td>27.279790</td>
      <td>30.962190</td>
      <td>21.186491</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2018</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>42.390026</td>
      <td>50.738111</td>
      <td>27.892743</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2019</td>
      <td>Licensed Practical Nurse</td>
      <td>27.511184</td>
      <td>31.190499</td>
      <td>21.094983</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2019</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>42.512560</td>
      <td>51.166334</td>
      <td>27.893060</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2020</td>
      <td>Licensed Practical Nurse</td>
      <td>28.015413</td>
      <td>31.304377</td>
      <td>22.707380</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2020</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>42.919098</td>
      <td>51.676908</td>
      <td>28.289764</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2021</td>
      <td>Licensed Practical Nurse</td>
      <td>28.643034</td>
      <td>32.124255</td>
      <td>24.203180</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2021</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>43.436805</td>
      <td>51.299163</td>
      <td>30.314302</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2022</td>
      <td>Licensed Practical Nurse</td>
      <td>30.192947</td>
      <td>33.873090</td>
      <td>24.739998</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2022</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>43.000474</td>
      <td>49.279176</td>
      <td>31.069909</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2023</td>
      <td>Licensed Practical Nurse</td>
      <td>31.801916</td>
      <td>33.376629</td>
      <td>24.076282</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2023</td>
      <td>Nurse Practitioners</td>
      <td>53.227121</td>
      <td>61.413847</td>
      <td>36.199318</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2023</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>44.297870</td>
      <td>52.169625</td>
      <td>32.437149</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-31d42687-b994-49e0-95ab-193b5bc56eb0')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-31d42687-b994-49e0-95ab-193b5bc56eb0 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-31d42687-b994-49e0-95ab-193b5bc56eb0');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-787a1da9-30ab-4ca6-9c9f-e90beda11059">
  <button class="colab-df-quickchart" onclick="quickchart('df-787a1da9-30ab-4ca6-9c9f-e90beda11059')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-787a1da9-30ab-4ca6-9c9f-e90beda11059 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_2abc48a8-8a20-4b37-8025-a23c421329a3">
    <style>
      .colab-df-generate {
        background-color: #E8F0FE;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        display: none;
        fill: #1967D2;
        height: 32px;
        padding: 0 0 0 0;
        width: 32px;
      }

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('occ_salary')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_2abc48a8-8a20-4b37-8025-a23c421329a3 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('occ_salary');
      }
      })();
    </script>
  </div>

    </div>
  </div>





```python
fig=px.scatter(occ_salary,x='Year',y='MEDIAN_WAGE_SALAIRE_MEDIAN',color='Occupation',symbol='Occupation',hover_name='Occupation')
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="72eea49a-b93c-49f3-8f76-f44f26f5a29d" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("72eea49a-b93c-49f3-8f76-f44f26f5a29d")) {                    Plotly.newPlot(                        "72eea49a-b93c-49f3-8f76-f44f26f5a29d",                        [{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eOccupation=Head Nurses and Supervisors\u003cbr\u003eYear=%{x}\u003cbr\u003eMEDIAN_WAGE_SALAIRE_MEDIAN=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Head Nurses and Supervisors","Head Nurses and Supervisors"],"legendgroup":"Head Nurses and Supervisors","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"Head Nurses and Supervisors","orientation":"v","showlegend":true,"x":[2014,2015],"xaxis":"x","y":[37.59618181818182,38.574090909090906],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eOccupation=Licensed Practical Nurse\u003cbr\u003eYear=%{x}\u003cbr\u003eMEDIAN_WAGE_SALAIRE_MEDIAN=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Licensed Practical Nurse","Licensed Practical Nurse","Licensed Practical Nurse","Licensed Practical Nurse","Licensed Practical Nurse","Licensed Practical Nurse","Licensed Practical Nurse","Licensed Practical Nurse","Licensed Practical Nurse","Licensed Practical Nurse"],"legendgroup":"Licensed Practical Nurse","marker":{"color":"#EF553B","symbol":"diamond"},"mode":"markers","name":"Licensed Practical Nurse","orientation":"v","showlegend":true,"x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"xaxis":"x","y":[25.244919913419913,25.389027805527803,25.96713311688312,26.933805194805192,27.279790043290042,27.511183982683985,28.015413419913422,28.643033910533912,30.192946778711487,31.80191575091575],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eOccupation=Registered Nurse and Registered Psych. Nurse\u003cbr\u003eYear=%{x}\u003cbr\u003eMEDIAN_WAGE_SALAIRE_MEDIAN=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Registered Nurse and Registered Psych. Nurse","Registered Nurse and Registered Psych. Nurse","Registered Nurse and Registered Psych. Nurse","Registered Nurse and Registered Psych. Nurse","Registered Nurse and Registered Psych. Nurse","Registered Nurse and Registered Psych. Nurse","Registered Nurse and Registered Psych. Nurse","Registered Nurse and Registered Psych. Nurse","Registered Nurse and Registered Psych. Nurse","Registered Nurse and Registered Psych. Nurse"],"legendgroup":"Registered Nurse and Registered Psych. Nurse","marker":{"color":"#00cc96","symbol":"square"},"mode":"markers","name":"Registered Nurse and Registered Psych. Nurse","orientation":"v","showlegend":true,"x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"xaxis":"x","y":[39.19123259303722,39.834801445578236,40.33560922562358,40.647006055452486,42.3900260770975,42.51256009070295,42.919098072562356,43.43680498866213,43.00047392290249,44.29787018140589],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eOccupation=Nurse Practitioners\u003cbr\u003eYear=%{x}\u003cbr\u003eMEDIAN_WAGE_SALAIRE_MEDIAN=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Nurse Practitioners"],"legendgroup":"Nurse Practitioners","marker":{"color":"#ab63fa","symbol":"x"},"mode":"markers","name":"Nurse Practitioners","orientation":"v","showlegend":true,"x":[2023],"xaxis":"x","y":[53.22712121212121],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Year"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"MEDIAN_WAGE_SALAIRE_MEDIAN"}},"legend":{"title":{"text":"Occupation"},"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('72eea49a-b93c-49f3-8f76-f44f26f5a29d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```python
workforce_total=workforce_rn[workforce_rn.Jurisdiction== 'Provinces/territories with available data'][['Year','Jurisdiction','Type of professional','Workforce: number \nof nurses']]
occ_no=workforce_total.groupby(['Year','Type of professional'])['Workforce: number \nof nurses'].sum().reset_index()
```


```python
occ_no
```


```python
fig=px.scatter(occ_no,x='Year',y='Workforce: number \nof nurses',color='Type of professional',symbol='Type of professional',
               hover_data='Workforce: number \nof nurses')

fig.show()
```


```python
series_d
```


```python
series_d.rename(columns={'Year ':'Year'},inplace=True)
series_d.Year=series_d.Year.apply(lambda x: int(x.split('–')[0]))

```


```python
series_d['Total_beds']=series[[x for x in series_d.columns.tolist() if 'beds' in x]].
```


```python
for col in [x for x in series_d.columns.tolist() if 'beds' in x]:
  series_d[col]=pd.to_numeric(series_d[col],errors='coerce')
series_d['Total_beds']=series_d[[x for x in series_d.columns.tolist() if 'beds' in x]].sum(axis=1)
series_d
```


```python
yearly_beds=series_d.groupby(['Year'])['Total_beds'].sum().reset_index()
yearly_fte=series_e.groupby(['Year'])['FTE_Total'].sum().reset_index()
```


```python
yearly_beds
```


```python
yearly_fte
```

# Job Vacancies in Nursing Professionals Data Collection and Cleaning


```python
!pip install stats-can
```


```python
from stats_can import StatsCan
import zipfile
import pandas as pd
sc =StatsCan()

job_vacancy=sc.table_to_df('14-10-0443-01')


nurses=['Registered nurses and registered psychiatric nurses [31301]','Nurse practitioners [31302]','Licensed practical nurses [32101]']

zip_f=zipfile.ZipFile('/content/14100443-eng.zip')
zip_f.extract(zip_f.infolist()[0])
```


```python
nurses=['Registered nurses and registered psychiatric nurses [31301]','Nurse practitioners [31302]','Licensed practical nurses [32101]']
import zipfile
import pandas as pd

zip_f=zipfile.ZipFile('/content/14100443-eng.zip')
zip_f.extract(zip_f.infolist()[0])
```

Note: The file size for the vacancies is quite large (~ 11 GB excel doc)


```python
nurses=['Registered nurses and registered psychiatric nurses','Nurse practitioners','Licensed practical nurses']
import pandas as pd
import pandas as pd

chunks=pd.read_csv('/content/14100443.csv',iterator=True,chunksize=1000000)
filtered_chunks=[]
for chunk in chunks:
  if 'National Occupational Classification' in chunk.columns:
    filtered_chunk= chunk[chunk['National Occupational Classification'].str.contains('|'.join(nurses),case=False,na=False)]
    filtered_chunks.append(filtered_chunk)
  else:
    print(chunk.columns)
    print('NOC not in columns')

job_vacancy=pd.concat(filtered_chunks,ignore_index= True)
```


```python
job_vacancy_can=job_vacancy[job_vacancy.GEO =='Canada']
job_vacancy_can
```


```python
job_vacancy_can['Statistics'].unique()
```


```python
job_vacancy_can['Job vacancy characteristics'].unique()
```


```python
job_vacancy_can[(job_vacancy_can['Job vacancy characteristics']=='Full-time') &
                (job_vacancy_can['Statistics']=='Job vacancies')][['REF_DATE','National Occupational Classification','VALUE']]
```


```python
job_vacancy_can['Year']=job_vacancy_can['REF_DATE'].apply(lambda x: int(x.split('-')[0]))
job_vacancy_can
```


```python
total_vacancy=job_vacancy_can[job_vacancy_can['Job vacancy characteristics']=='Type of work, all types']
vac_count=total_vacancy.groupby(['Year','National Occupational Classification'])['VALUE'].sum().reset_index()
```


```python
import plotly.express as px


fig=px.line(vac_count,x='Year',y='VALUE',color='National Occupational Classification',
               symbol='National Occupational Classification',hover_name='National Occupational Classification',
            title='Healthcare Vacancies in Canada 2015-24')
fig.update_yaxes(title_text='Total Vacancies')
fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="24fab042-5824-41bb-ae78-b821e745e405" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("24fab042-5824-41bb-ae78-b821e745e405")) {                    Plotly.newPlot(                        "24fab042-5824-41bb-ae78-b821e745e405",                        [{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eNational Occupational Classification=Licensed practical nurses [32101]\u003cbr\u003eYear=%{x}\u003cbr\u003eVALUE=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Licensed practical nurses [32101]","Licensed practical nurses [32101]","Licensed practical nurses [32101]","Licensed practical nurses [32101]","Licensed practical nurses [32101]","Licensed practical nurses [32101]","Licensed practical nurses [32101]","Licensed practical nurses [32101]","Licensed practical nurses [32101]","Licensed practical nurses [32101]"],"legendgroup":"Licensed practical nurses [32101]","line":{"color":"#636efa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines+markers","name":"Licensed practical nurses [32101]","orientation":"v","showlegend":true,"x":[2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],"xaxis":"x","y":[7068.4,7284.65,12387.55,13943.85,15664.85,11830.75,38815.55,48834.25,54147.7,25333.8],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eNational Occupational Classification=Nurse practitioners [31302]\u003cbr\u003eYear=%{x}\u003cbr\u003eVALUE=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Nurse practitioners [31302]","Nurse practitioners [31302]","Nurse practitioners [31302]","Nurse practitioners [31302]","Nurse practitioners [31302]","Nurse practitioners [31302]","Nurse practitioners [31302]","Nurse practitioners [31302]","Nurse practitioners [31302]","Nurse practitioners [31302]"],"legendgroup":"Nurse practitioners [31302]","line":{"color":"#EF553B","dash":"solid"},"marker":{"symbol":"diamond"},"mode":"lines+markers","name":"Nurse practitioners [31302]","orientation":"v","showlegend":true,"x":[2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],"xaxis":"x","y":[728.05,799.0,1079.75,1554.75,1642.3,868.4,2210.65,2599.8,3494.7,2193.25],"yaxis":"y","type":"scatter"},{"hovertemplate":"\u003cb\u003e%{hovertext}\u003c\u002fb\u003e\u003cbr\u003e\u003cbr\u003eNational Occupational Classification=Registered nurses and registered psychiatric nurses [31301]\u003cbr\u003eYear=%{x}\u003cbr\u003eVALUE=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","hovertext":["Registered nurses and registered psychiatric nurses [31301]","Registered nurses and registered psychiatric nurses [31301]","Registered nurses and registered psychiatric nurses [31301]","Registered nurses and registered psychiatric nurses [31301]","Registered nurses and registered psychiatric nurses [31301]","Registered nurses and registered psychiatric nurses [31301]","Registered nurses and registered psychiatric nurses [31301]","Registered nurses and registered psychiatric nurses [31301]","Registered nurses and registered psychiatric nurses [31301]","Registered nurses and registered psychiatric nurses [31301]"],"legendgroup":"Registered nurses and registered psychiatric nurses [31301]","line":{"color":"#00cc96","dash":"solid"},"marker":{"symbol":"square"},"mode":"lines+markers","name":"Registered nurses and registered psychiatric nurses [31301]","orientation":"v","showlegend":true,"x":[2015,2016,2017,2018,2019,2020,2021,2022,2023,2024],"xaxis":"x","y":[19638.75,30316.5,31412.5,40840.950000000004,45853.45,34960.7,89389.15,102408.15,117570.0,56634.700000000004],"yaxis":"y","type":"scatter"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Year"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Total Vacancies"}},"legend":{"title":{"text":"National Occupational Classification"},"tracegroupgap":0},"title":{"text":"Healthcare Vacancies in Canada 2015-24"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('24fab042-5824-41bb-ae78-b821e745e405');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


# Nursing Canada 2014-23 Data Collection & Cleaning


```python
can_vac=vac_count.groupby(['Year'])['VALUE'].sum().reset_index()
can_vac
```


```python
supply_rn=pd.read_excel('/content/nursing-in-canada-2014-2023-data-tables-en.xlsx',sheet_name='4 Supply',skiprows=1,nrows=460)
supply_rn
```


```python
import numpy as np
supply_rn['Year']=supply_rn['Year'].astype('int')
supply_rn['Supply: \nnumber \nof nurses']=supply_rn['Supply: \nnumber \nof nurses'].astype(str).replace('-',np.nan)
supply_rn['Supply: \nnumber \nof nurses']=pd.to_numeric(supply_rn['Supply: \nnumber \nof nurses'],errors='coerce')
supply_rn
```


```python
supply_rn
can_supply=supply_rn.groupby(['Year'])['Supply: \nnumber \nof nurses'].sum().reset_index()
can_supply.rename(columns={'Supply: \nnumber \nof nurses':'Nurse Supply'},inplace=True)
can_supply
```


```python
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig=make_subplots(specs=[[{'secondary_y':True}]])

x=can_supply.Year
y1=can_supply['Nurse Supply']
y2=can_vac.VALUE

fig.add_trace(go.Scatter(x=x,y=y1,mode='lines+markers',name='Supply'),secondary_y=False)
fig.add_trace(go.Scatter(x=x,y=y2,mode='lines+markers',name='Vacancy'),secondary_y=True)

fig.update_yaxes(title_text='Total Supply of Professionals',secondary_y=False)
fig.update_yaxes(title_text='Total Vacancies',secondary_y=True)
fig.update_xaxes(title_text='Year')

fig.show()
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="d85d3596-d915-4410-a503-26f3e75dea70" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("d85d3596-d915-4410-a503-26f3e75dea70")) {                    Plotly.newPlot(                        "d85d3596-d915-4410-a503-26f3e75dea70",                        [{"mode":"lines+markers","name":"Supply","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[814002.0,832134.0,842652.0,852090.0,863854.0,879694.0,896668.0,916542.0,932028.0,955960.0],"type":"scatter","xaxis":"x","yaxis":"y"},{"mode":"lines+markers","name":"Vacancy","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[27435.2,38400.15,44879.8,56339.55,63160.6,47659.85,130415.35,153842.2,175212.4,84161.75],"type":"scatter","xaxis":"x","yaxis":"y2"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.94],"title":{"text":"Year"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Total Supply of Professionals"}},"yaxis2":{"anchor":"x","overlaying":"y","side":"right","title":{"text":"Total Vacancies"}}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('d85d3596-d915-4410-a503-26f3e75dea70');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```python
supply_rn
```


```python
supply_rn.columns
```


```python
supply_in=supply_rn[['Year','Jurisdiction','Type of professional','Supply: \ninflow']]
supply_out=supply_rn[['Year','Jurisdiction','Type of professional','Supply: \noutflow ']]

supply_in['Year']=supply_in['Year'].astype('int')
supply_out['Year']=supply_out['Year'].astype('int')
supply_in['Supply: \ninflow']=pd.to_numeric(supply_in['Supply: \ninflow'].replace('-',0),errors='coerce')
supply_out['Supply: \noutflow ']=pd.to_numeric(supply_out['Supply: \noutflow '].replace('-',0),errors='coerce')

can_in=supply_in.groupby(['Year'])['Supply: \ninflow'].sum().reset_index()
can_out=supply_out.groupby(['Year'])['Supply: \noutflow '].sum().reset_index()
can_in.rename(columns={'Supply: \ninflow':'Inflow'},inplace=True)
can_out.rename(columns={'Supply: \noutflow ':'Outflow'},inplace=True)


fig=make_subplots()

x=can_in.Year
y1=can_in.Inflow
y2=can_out.Outflow

fig.add_trace(go.Scatter(x=x,y=y1,name='Inflow',mode='lines+markers'))
fig.add_trace(go.Scatter(x=x,y=y2,name='Outflow',mode='lines+markers'))

fig.update_yaxes(title_text='Inflow/Outflow')
fig.update_xaxes(title_text='Year')
fig.update_layout(title_text='Inflow/Outflow of Nurses in Canada')

fig.show()
```


```python
from stats_can import StatsCan
sc =StatsCan()

can_population=sc.table_to_df('17-10-0009-01')
can_population=can_population[can_population.GEO == 'Canada']
can_population['Year']=can_population.REF_DATE.apply(lambda x: x.year)
can_pop_count=can_population[can_population.Year>=2014]

import pandas as pd
can_pop={'Year':[],'Population':[]}

for y in can_pop_count.Year.unique():
  can_pop['Year'].append(y)
  can_pop['Population'].append(can_pop_count[can_pop_count['Year']==y]['VALUE'].max())

can_pop=pd.DataFrame(can_pop)
can_pop
```


```python
import pandas as pd
import numpy as np

nurse_data=pd.read_excel('/content/nursing-in-canada-2014-2023-data-tables-en.xlsx',
                         sheet_name='7 Emp dir care per pop',skiprows=1,nrows=40)
```


```python
nurse_data['Year']=nurse_data['Year'].apply(lambda x: int(x.replace('*','')) if x=='2023*' else int(x))

nurse_data['Provinces/territories with available data']=pd.to_numeric(nurse_data['Provinces/territories with available data'],errors='coerce')
```


```python
professional_needs=nurse_data[['Year','Type of professional','Provinces/territories with available data']]

nurse_pop=pd.merge(professional_needs,can_pop,on='Year')
nurse_pop
```


```python
nurse_pop['Professionals Required']=(nurse_pop['Provinces/territories with available data']*nurse_pop['Population'])/100000
can_needs=nurse_pop.groupby(['Year'])['Professionals Required'].sum().reset_index()
can_needs
```


```python
workforce_rn=pd.read_excel('/content/nursing-in-canada-2014-2023-data-tables-en.xlsx',sheet_name='5 Workforce',skiprows=1,nrows=460)
```


```python
workforce_rn['Year']=workforce_rn['Year'].astype('int')
workforce_rn['Workforce: number \nof nurses']=pd.to_numeric(workforce_rn['Workforce: number \nof nurses'].astype(str).replace('-',np.nan),
                                                             errors='coerce')
can_workforce=workforce_rn.groupby(['Year'])['Workforce: number \nof nurses'].sum().reset_index()
can_workforce.rename(columns={'Workforce: number \nof nurses':'Workforce Total'},inplace=True)
can_workforce
```


```python
nurses_direct_supply=pd.read_excel('/content/nursing-in-canada-2014-2023-data-tables-en.xlsx',sheet_name='9 Ratio emp dir care',
                            skiprows=1,nrows=40)
nurses_direct_supply
```


```python
nurses_direct_supply['Year']=nurses_direct_supply['Year'].astype('int')
nurses_direct_supply['Employed in \ndirect care']=pd.to_numeric(nurses_direct_supply['Employed in \ndirect care'].replace('-',np.nan),errors='coerce')
nurses_direct_supply.rename(columns={'Employed in \ndirect care':'Direct Care'},inplace=True)
sum_direct=nurses_direct_supply.groupby(['Year'])['Direct Care'].sum().reset_index()
```


```python
sum_direct
```


```python
nurses_direct_pop=pd.read_excel('/content/nursing-in-canada-2014-2023-data-tables-en.xlsx',sheet_name='12 Emp dir care settings',
                            skiprows=2,nrows=40)
nurses_direct_pop
```


```python
nurses_direct_pop['Year']=nurses_direct_pop['Year'].astype('int')

for col in nurses_direct_pop.columns[1:]:
  if '%' not in col:
    nurses_direct_pop[col]=pd.to_numeric(nurses_direct_pop[col].replace('-',np.nan),errors='coerce')

```


```python
num_columns=[x for x in nurses_direct_pop.columns if '%' not in x]
nurses_direct_pop['Total']=nurses_direct_pop[num_columns[2:]].sum(axis=1)
count_direct_pop=nurses_direct_pop.groupby(['Year'])['Total'].sum().reset_index()
count_direct_pop
```


```python
nurses_direct_pop[['Type of professional','Year','Total']]
nurses_direct_pop.groupby(['Type of professional','Year'])['Total'].sum().reset_index()
```


```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig2=make_subplots(specs=[[{"secondary_y":True}]])

x=can_needs.Year
y1=can_needs['Professionals Required']
y2=sum_direct['Direct Care']
y3=can_pop['Population']

fig2.add_trace(go.Scatter(x=x,y=y1,mode='lines+markers',name='Professional Demanded'),secondary_y=False)
fig2.add_trace(go.Scatter(x=x,y=y2,mode='lines+markers',name='Professional Supplied'),secondary_y=False)
fig2.add_trace(go.Scatter(x=x,y=y3,mode='lines+markers',name='Population'),secondary_y=True)

fig2.update_yaxes(title_text='Nurse Needs/Supply',secondary_y=False)
fig2.update_yaxes(title_text='Population',secondary_y=True)
fig2.update_xaxes(title_text='Year')
fig2.add_vline(x=2020,annotation_text='COVID-19',annotation_position='top left')
fig2.update_layout(title_text='Direct Care Nurse Needs/Supply in Canada')
```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="4f5e93af-808e-422d-b419-cc43d0addc2a" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("4f5e93af-808e-422d-b419-cc43d0addc2a")) {                    Plotly.newPlot(                        "4f5e93af-808e-422d-b419-cc43d0addc2a",                        [{"mode":"lines+markers","name":"Professional Demanded","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[359393.02294,365328.981018,371529.79298699996,378017.04005,381201.791035,371434.722678,394052.746956,404883.998501,411374.21397299995,430904.574716],"type":"scatter","xaxis":"x","yaxis":"y"},{"mode":"lines+markers","name":"Professional Supplied","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[347653.0,353770.0,359535.0,365484.0,368684.0,357199.0,372913.0,379597.0,384593.0,391729.0],"type":"scatter","xaxis":"x","yaxis":"y"},{"mode":"lines+markers","name":"Population","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[35555305.0,35823591.0,36257421.0,36722075.0,37259485.0,37828162.0,38028638.0,38446871.0,39279501.0,40513781.0,41288599.0],"type":"scatter","xaxis":"x","yaxis":"y2"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.94],"title":{"text":"Year"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Nurse Needs\u002fSupply"}},"yaxis2":{"anchor":"x","overlaying":"y","side":"right","title":{"text":"Population"}},"shapes":[{"type":"line","x0":2020,"x1":2020,"xref":"x","y0":0,"y1":1,"yref":"y domain"}],"annotations":[{"showarrow":false,"text":"COVID-19","x":2020,"xanchor":"right","xref":"x","y":1,"yanchor":"top","yref":"y domain"}],"title":{"text":"Direct Care Nurse Needs\u002fSupply in Canada"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('4f5e93af-808e-422d-b419-cc43d0addc2a');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>



```python
can_workforce_needs=pd.merge(can_workforce,can_needs,on='Year')
can_proff_brk=pd.merge(can_workforce_needs,sum_direct,on='Year')
can_proff_brk['Non-Direct Care']=can_proff_brk['Workforce Total']-can_proff_brk['Direct Care']
```


```python
can_proff_brk
```

# Nursing Hours Worked 2014-23


```python
!pip install stats_can
```


```python
from stats_can import StatsCan

sc=StatsCan()
can_work_hrs=sc.table_to_df('14-10-0423-01')
can_work_hrs
```


```python
nurse_work=can_work_hrs[can_work_hrs.GEO=='Canada']
nurse_work=nurse_work[(nurse_work['National Occupational Classification (NOC)']=='Nursing and allied health professionals [313]') &
                      (nurse_work.Sex == 'Both sexes') &
                      (nurse_work['Actual hours worked']=='Total employed, all hours')]
nurse_work
```


```python
nurse_work['Year']=nurse_work.REF_DATE.apply(lambda x: x.year)
```


```python
can_nurse_work=nurse_work.groupby(['Year'])['VALUE'].sum().reset_index()
can_nurse_work.rename(columns={'VALUE':'hours_worked_pp'},inplace=True)
can_nurse_work=can_nurse_work[can_nurse_work.Year.isin(range(2014,2024))]
can_nurse_work
```


```python
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig=make_subplots(specs=[[{"secondary_y": True}]])

x=can_nurse_work.Year
y1=can_nurse_work.hours_worked_pp
y2=yearly_sal.LOW_WAGE_SALAIRE_MINIUM
y3=yearly_sal.MEDIAN_WAGE_SALAIRE_MEDIAN
y4=yearly_sal.HIGH_WAGE_SALAIRE_MAXIMAL

fig.add_trace(go.Scatter(x=x,y=y1,mode='lines+markers',name='Hours Worked PP'),secondary_y=False)
fig.add_trace(go.Scatter(x=x,y=y2,name='Min Hourly Salary',mode='markers'),secondary_y=True)
fig.add_trace(go.Scatter(x=x,y=y3,name='Median Hourly Salary',mode='lines+markers'),secondary_y=True)
fig.add_trace(go.Scatter(x=x,y=y4,name='Max Hourly Salary',mode='markers'),secondary_y=True)

shapes=[go.layout.Shape(type='line',
                        xref='x',
                        yref='y2',
                        x0=i['Year'],
                        x1=i['Year'],
                        y0=i['LOW_WAGE_SALAIRE_MINIUM'],
                        y1=i['HIGH_WAGE_SALAIRE_MAXIMAL'],
                        x0shift=1,
                        opacity=0.5
                    ) for r,i in yearly_sal.iterrows()]

fig.update_layout(shapes=shapes)


fig.update_yaxes(title_text='Hourse Worked Per Person',secondary_y=False)
fig.update_yaxes(title_text='Median Hourly Salary',secondary_y=True)
fig.update_xaxes(title_text='Year')
fig.add_vrect(x0=2019+(11/12),x1=2022+(5/12),annotation_text='COVID-19',annotation_position='top',
              annotation=dict(font_size=12),fillcolor='red',opacity=0.3)

fig.update_layout(title_text='Hours Worked Per Person vs Median Hourly Salary')
fig.show()

```


<html>
<head><meta charset="utf-8" /></head>
<body>
    <div>            <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS-MML_SVG"></script><script type="text/javascript">if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script>                <script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script>
        <script charset="utf-8" src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>                <div id="825f25e5-62e9-47cc-b7da-40a825aa650e" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("825f25e5-62e9-47cc-b7da-40a825aa650e")) {                    Plotly.newPlot(                        "825f25e5-62e9-47cc-b7da-40a825aa650e",                        [{"mode":"lines+markers","name":"Hours Worked PP","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[3834.1,3953.7000000000003,4099.0,4077.0,4252.6,4190.8,4190.4,4332.6,4705.8,4837.2],"type":"scatter","xaxis":"x","yaxis":"y"},{"mode":"markers","name":"Min Hourly Salary","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[21.067040129940864,21.281009827363995,23.686018493333336,24.626508347763348,24.941992222222222,24.901905714285714,25.833514920634922,27.625408412698413,28.02217446484767,30.72803076988371],"type":"scatter","xaxis":"x","yaxis":"y2"},{"mode":"lines+markers","name":"Median Hourly Salary","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[34.442482648615005,35.0355979471917,34.01347973777778,34.613197676767676,35.74152222222222,35.91195460317461,36.36147682539683,36.92754571428571,36.83388677940312,42.60772159565581],"type":"scatter","xaxis":"x","yaxis":"y2"},{"mode":"markers","name":"Max Hourly Salary","x":[2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],"y":[42.69919238575059,41.41944948531407,40.18626751428572,40.969561688311686,42.036705714285716,42.37696666666667,42.71299428571429,42.86220365079365,41.861431147767746,48.53653787878788],"type":"scatter","xaxis":"x","yaxis":"y2"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.94],"title":{"text":"Year"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Hourse Worked Per Person"}},"yaxis2":{"anchor":"x","overlaying":"y","side":"right","title":{"text":"Median Hourly Salary"}},"shapes":[{"opacity":0.5,"type":"line","x0":2014.0,"x0shift":1,"x1":2014.0,"xref":"x","y0":21.067040129940864,"y1":42.69919238575059,"yref":"y2"},{"opacity":0.5,"type":"line","x0":2015.0,"x0shift":1,"x1":2015.0,"xref":"x","y0":21.281009827363995,"y1":41.41944948531407,"yref":"y2"},{"opacity":0.5,"type":"line","x0":2016.0,"x0shift":1,"x1":2016.0,"xref":"x","y0":23.686018493333336,"y1":40.18626751428572,"yref":"y2"},{"opacity":0.5,"type":"line","x0":2017.0,"x0shift":1,"x1":2017.0,"xref":"x","y0":24.626508347763348,"y1":40.969561688311686,"yref":"y2"},{"opacity":0.5,"type":"line","x0":2018.0,"x0shift":1,"x1":2018.0,"xref":"x","y0":24.941992222222222,"y1":42.036705714285716,"yref":"y2"},{"opacity":0.5,"type":"line","x0":2019.0,"x0shift":1,"x1":2019.0,"xref":"x","y0":24.901905714285714,"y1":42.37696666666667,"yref":"y2"},{"opacity":0.5,"type":"line","x0":2020.0,"x0shift":1,"x1":2020.0,"xref":"x","y0":25.833514920634922,"y1":42.71299428571429,"yref":"y2"},{"opacity":0.5,"type":"line","x0":2021.0,"x0shift":1,"x1":2021.0,"xref":"x","y0":27.625408412698413,"y1":42.86220365079365,"yref":"y2"},{"opacity":0.5,"type":"line","x0":2022.0,"x0shift":1,"x1":2022.0,"xref":"x","y0":28.02217446484767,"y1":41.861431147767746,"yref":"y2"},{"opacity":0.5,"type":"line","x0":2023.0,"x0shift":1,"x1":2023.0,"xref":"x","y0":30.72803076988371,"y1":48.53653787878788,"yref":"y2"},{"fillcolor":"red","opacity":0.3,"type":"rect","x0":2019.9166666666667,"x1":2022.4166666666667,"xref":"x","y0":0,"y1":1,"yref":"y domain"}],"annotations":[{"font":{"size":12},"showarrow":false,"text":"COVID-19","x":2021.1666666666667,"xanchor":"center","xref":"x","y":1,"yanchor":"top","yref":"y domain"}],"title":{"text":"Hours Worked Per Person vs Median Hourly Salary"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('825f25e5-62e9-47cc-b7da-40a825aa650e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                            </script>        </div>
</body>
</html>


# Dash Application for Nursing Dashboard


```python
!pip install dash
!pip install stats-can
import dash
from dash import Dash, dcc, html, Input, Output,callback
import plotly.express as px
import pandas as pd
import numpy as np
from stats_can import StatsCan
```

    Requirement already satisfied: dash in c:\users\user\anaconda3\lib\site-packages (2.18.2)
    Requirement already satisfied: Flask<3.1,>=1.0.4 in c:\users\user\anaconda3\lib\site-packages (from dash) (2.2.5)
    Requirement already satisfied: Werkzeug<3.1 in c:\users\user\anaconda3\lib\site-packages (from dash) (2.2.3)
    Requirement already satisfied: plotly>=5.0.0 in c:\users\user\anaconda3\lib\site-packages (from dash) (5.9.0)
    Requirement already satisfied: dash-html-components==2.0.0 in c:\users\user\anaconda3\lib\site-packages (from dash) (2.0.0)
    Requirement already satisfied: dash-core-components==2.0.0 in c:\users\user\anaconda3\lib\site-packages (from dash) (2.0.0)
    Requirement already satisfied: dash-table==5.0.0 in c:\users\user\anaconda3\lib\site-packages (from dash) (5.0.0)
    Requirement already satisfied: importlib-metadata in c:\users\user\anaconda3\lib\site-packages (from dash) (7.0.1)
    Requirement already satisfied: typing-extensions>=4.1.1 in c:\users\user\anaconda3\lib\site-packages (from dash) (4.9.0)
    Requirement already satisfied: requests in c:\users\user\anaconda3\lib\site-packages (from dash) (2.32.3)
    Requirement already satisfied: retrying in c:\users\user\anaconda3\lib\site-packages (from dash) (1.3.4)
    Requirement already satisfied: nest-asyncio in c:\users\user\anaconda3\lib\site-packages (from dash) (1.6.0)
    Requirement already satisfied: setuptools in c:\users\user\anaconda3\lib\site-packages (from dash) (68.2.2)
    Requirement already satisfied: Jinja2>=3.0 in c:\users\user\anaconda3\lib\site-packages (from Flask<3.1,>=1.0.4->dash) (3.1.3)
    Requirement already satisfied: itsdangerous>=2.0 in c:\users\user\anaconda3\lib\site-packages (from Flask<3.1,>=1.0.4->dash) (2.0.1)
    Requirement already satisfied: click>=8.0 in c:\users\user\anaconda3\lib\site-packages (from Flask<3.1,>=1.0.4->dash) (8.1.7)
    Requirement already satisfied: tenacity>=6.2.0 in c:\users\user\anaconda3\lib\site-packages (from plotly>=5.0.0->dash) (8.2.2)
    Requirement already satisfied: MarkupSafe>=2.1.1 in c:\users\user\anaconda3\lib\site-packages (from Werkzeug<3.1->dash) (2.1.3)
    Requirement already satisfied: zipp>=0.5 in c:\users\user\anaconda3\lib\site-packages (from importlib-metadata->dash) (3.17.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\user\anaconda3\lib\site-packages (from requests->dash) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\user\anaconda3\lib\site-packages (from requests->dash) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\user\anaconda3\lib\site-packages (from requests->dash) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\user\anaconda3\lib\site-packages (from requests->dash) (2024.8.30)
    Requirement already satisfied: six>=1.7.0 in c:\users\user\anaconda3\lib\site-packages (from retrying->dash) (1.16.0)
    Requirement already satisfied: colorama in c:\users\user\anaconda3\lib\site-packages (from click>=8.0->Flask<3.1,>=1.0.4->dash) (0.4.6)
    Requirement already satisfied: stats-can in c:\users\user\anaconda3\lib\site-packages (2.9.4)
    Requirement already satisfied: h5py in c:\users\user\anaconda3\lib\site-packages (from stats-can) (3.11.0)
    Requirement already satisfied: numpy in c:\users\user\anaconda3\lib\site-packages (from stats-can) (1.26.4)
    Requirement already satisfied: pandas in c:\users\user\anaconda3\lib\site-packages (from stats-can) (2.1.4)
    Requirement already satisfied: requests in c:\users\user\anaconda3\lib\site-packages (from stats-can) (2.32.3)
    Requirement already satisfied: tables in c:\users\user\anaconda3\lib\site-packages (from stats-can) (3.9.2)
    Requirement already satisfied: tqdm in c:\users\user\anaconda3\lib\site-packages (from stats-can) (4.65.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\user\anaconda3\lib\site-packages (from pandas->stats-can) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\users\user\anaconda3\lib\site-packages (from pandas->stats-can) (2023.3.post1)
    Requirement already satisfied: tzdata>=2022.1 in c:\users\user\anaconda3\lib\site-packages (from pandas->stats-can) (2023.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\user\anaconda3\lib\site-packages (from requests->stats-can) (2.0.4)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\user\anaconda3\lib\site-packages (from requests->stats-can) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\user\anaconda3\lib\site-packages (from requests->stats-can) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\user\anaconda3\lib\site-packages (from requests->stats-can) (2024.8.30)
    Requirement already satisfied: numexpr>=2.6.2 in c:\users\user\anaconda3\lib\site-packages (from tables->stats-can) (2.8.7)
    Requirement already satisfied: packaging in c:\users\user\anaconda3\lib\site-packages (from tables->stats-can) (23.1)
    Requirement already satisfied: py-cpuinfo in c:\users\user\anaconda3\lib\site-packages (from tables->stats-can) (9.0.0)
    Requirement already satisfied: colorama in c:\users\user\anaconda3\lib\site-packages (from tqdm->stats-can) (0.4.6)
    Requirement already satisfied: six>=1.5 in c:\users\user\anaconda3\lib\site-packages (from python-dateutil>=2.8.2->pandas->stats-can) (1.16.0)
    

## Reupload and Clean Datasets


```python

```


```python
supply_professionals=pd.read_excel('nursing-in-canada-2014-2023-data-tables-en.xlsx',sheet_name='4 Supply',skiprows=1,nrows=460)
workforce_professionals=pd.read_excel('nursing-in-canada-2014-2023-data-tables-en.xlsx',sheet_name='5 Workforce',skiprows=1,nrows=460)
```


```python
supply_professionals.drop([col for col in supply_professionals if 'Column' in col],axis=1,inplace=True)

```


```python
for col in workforce_professionals.columns:
    if 'Workforce' in col:
        workforce_professionals[col]=pd.to_numeric(workforce_professionals[col].replace('-',np.nan),errors='coerce')

for col in supply_professionals.columns:
    if 'Workforce' in col:
        supply_professionals[col]=pd.to_numeric(supply_professionals[col].replace('-',np.nan),errors='coerce')
```


```python
from stats_can import StatsCan

sc=StatsCan()
can_work_hrs=sc.table_to_df('14-10-0423-01')
can_work_hrs
#can_work_hrs=can_work_hrs[can_work_hrs.GEO=='Canada']
nur_work_hrs=can_work_hrs[(can_work_hrs['National Occupational Classification (NOC)']=='Nursing and allied health professionals [313]') &
                      (can_work_hrs.Sex == 'Both sexes') &
                      (can_work_hrs['Actual hours worked']=='Total actual hours (main job)')]
nur_work_hrs
```

    C:\Users\user\anaconda3\Lib\site-packages\stats_can\api_class.py:24: FutureWarning:
    
    This class will be deprecated in upcoming v3 release. Please see the docs for details
    
    C:\Users\user\anaconda3\Lib\site-packages\stats_can\sc.py:608: FutureWarning:
    
    This function will be deprecated in the v3 release. Please see the docs for details.
    
    C:\Users\user\anaconda3\Lib\site-packages\stats_can\sc.py:326: FutureWarning:
    
    This function will be deprecated in the v3 release. Please see the docs for details.
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REF_DATE</th>
      <th>GEO</th>
      <th>DGUID</th>
      <th>Actual hours worked</th>
      <th>National Occupational Classification (NOC)</th>
      <th>Sex</th>
      <th>UOM</th>
      <th>UOM_ID</th>
      <th>SCALAR_FACTOR</th>
      <th>SCALAR_ID</th>
      <th>VECTOR</th>
      <th>COORDINATE</th>
      <th>VALUE</th>
      <th>STATUS</th>
      <th>SYMBOL</th>
      <th>TERMINATED</th>
      <th>DECIMALS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1662</th>
      <td>1987-01-01</td>
      <td>Canada</td>
      <td>2016A000011124</td>
      <td>Total actual hours (main job)</td>
      <td>Nursing and allied health professionals [313]</td>
      <td>Both sexes</td>
      <td>Hours</td>
      <td>152</td>
      <td>units</td>
      <td>0</td>
      <td>v1489990018</td>
      <td>1.10.24.1</td>
      <td>6109.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3786</th>
      <td>1987-01-01</td>
      <td>Newfoundland and Labrador</td>
      <td>2016A000210</td>
      <td>Total actual hours (main job)</td>
      <td>Nursing and allied health professionals [313]</td>
      <td>Both sexes</td>
      <td>Hours</td>
      <td>152</td>
      <td>units</td>
      <td>0</td>
      <td>v1489990054</td>
      <td>2.10.24.1</td>
      <td>117.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5910</th>
      <td>1987-01-01</td>
      <td>Prince Edward Island</td>
      <td>2016A000211</td>
      <td>Total actual hours (main job)</td>
      <td>Nursing and allied health professionals [313]</td>
      <td>Both sexes</td>
      <td>Hours</td>
      <td>152</td>
      <td>units</td>
      <td>0</td>
      <td>v1489990090</td>
      <td>3.10.24.1</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8034</th>
      <td>1987-01-01</td>
      <td>Nova Scotia</td>
      <td>2016A000212</td>
      <td>Total actual hours (main job)</td>
      <td>Nursing and allied health professionals [313]</td>
      <td>Both sexes</td>
      <td>Hours</td>
      <td>152</td>
      <td>units</td>
      <td>0</td>
      <td>v1489990126</td>
      <td>4.10.24.1</td>
      <td>198.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10158</th>
      <td>1987-01-01</td>
      <td>New Brunswick</td>
      <td>2016A000213</td>
      <td>Total actual hours (main job)</td>
      <td>Nursing and allied health professionals [313]</td>
      <td>Both sexes</td>
      <td>Hours</td>
      <td>152</td>
      <td>units</td>
      <td>0</td>
      <td>v1489990162</td>
      <td>5.10.24.1</td>
      <td>226.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10598298</th>
      <td>2024-10-01</td>
      <td>Ontario</td>
      <td>2016A000235</td>
      <td>Total actual hours (main job)</td>
      <td>Nursing and allied health professionals [313]</td>
      <td>Both sexes</td>
      <td>Hours</td>
      <td>152</td>
      <td>units</td>
      <td>0</td>
      <td>v1489990234</td>
      <td>7.10.24.1</td>
      <td>4933.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10600422</th>
      <td>2024-10-01</td>
      <td>Manitoba</td>
      <td>2016A000246</td>
      <td>Total actual hours (main job)</td>
      <td>Nursing and allied health professionals [313]</td>
      <td>Both sexes</td>
      <td>Hours</td>
      <td>152</td>
      <td>units</td>
      <td>0</td>
      <td>v1489990270</td>
      <td>8.10.24.1</td>
      <td>468.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10602546</th>
      <td>2024-10-01</td>
      <td>Saskatchewan</td>
      <td>2016A000247</td>
      <td>Total actual hours (main job)</td>
      <td>Nursing and allied health professionals [313]</td>
      <td>Both sexes</td>
      <td>Hours</td>
      <td>152</td>
      <td>units</td>
      <td>0</td>
      <td>v1489990306</td>
      <td>9.10.24.1</td>
      <td>542.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10604670</th>
      <td>2024-10-01</td>
      <td>Alberta</td>
      <td>2016A000248</td>
      <td>Total actual hours (main job)</td>
      <td>Nursing and allied health professionals [313]</td>
      <td>Both sexes</td>
      <td>Hours</td>
      <td>152</td>
      <td>units</td>
      <td>0</td>
      <td>v1489990342</td>
      <td>10.10.24.1</td>
      <td>1500.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10606794</th>
      <td>2024-10-01</td>
      <td>British Columbia</td>
      <td>2016A000259</td>
      <td>Total actual hours (main job)</td>
      <td>Nursing and allied health professionals [313]</td>
      <td>Both sexes</td>
      <td>Hours</td>
      <td>152</td>
      <td>units</td>
      <td>0</td>
      <td>v1489990378</td>
      <td>11.10.24.1</td>
      <td>2159.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>4994 rows × 17 columns</p>
</div>




```python
can_work_hrs['Actual hours worked'].unique()
```




    array(['Total employed, all hours', '0 hours', '1 to 14 hours',
           '15 to 29 hours', '30 to 34 hours', '35 to 39 hours', '40 hours',
           '41 to 49 hours', '50 hours or more',
           'Total actual hours (main job)',
           'Average actual hours (all workers, main job)',
           'Average actual hours (worked in reference week, main job)'],
          dtype=object)




```python
nur_work_hrs['Year']=nur_work_hrs['REF_DATE'].apply(lambda x: int(x.year))
work_hrs=nur_work_hrs.groupby(['Year','GEO'])['VALUE'].sum().reset_index()
work_hrs=work_hrs[work_hrs.Year >=2014]
work_hrs
```

    C:\Users\user\AppData\Local\Temp\ipykernel_4644\1455252947.py:1: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    C:\Users\user\AppData\Local\Temp\ipykernel_4644\1455252947.py:2: FutureWarning:
    
    The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>GEO</th>
      <th>VALUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>297</th>
      <td>2014</td>
      <td>Alberta</td>
      <td>12506.3</td>
    </tr>
    <tr>
      <th>298</th>
      <td>2014</td>
      <td>British Columbia</td>
      <td>13058.8</td>
    </tr>
    <tr>
      <th>299</th>
      <td>2014</td>
      <td>Canada</td>
      <td>110220.0</td>
    </tr>
    <tr>
      <th>300</th>
      <td>2014</td>
      <td>Manitoba</td>
      <td>5391.0</td>
    </tr>
    <tr>
      <th>301</th>
      <td>2014</td>
      <td>New Brunswick</td>
      <td>2757.6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>2024</td>
      <td>Nova Scotia</td>
      <td>3994.9</td>
    </tr>
    <tr>
      <th>414</th>
      <td>2024</td>
      <td>Ontario</td>
      <td>47802.3</td>
    </tr>
    <tr>
      <th>415</th>
      <td>2024</td>
      <td>Prince Edward Island</td>
      <td>761.5</td>
    </tr>
    <tr>
      <th>416</th>
      <td>2024</td>
      <td>Quebec</td>
      <td>25665.8</td>
    </tr>
    <tr>
      <th>417</th>
      <td>2024</td>
      <td>Saskatchewan</td>
      <td>4173.2</td>
    </tr>
  </tbody>
</table>
<p>121 rows × 3 columns</p>
</div>




```python
work_hrs.GEO.unique().tolist()
```




    ['Alberta',
     'British Columbia',
     'Canada',
     'Manitoba',
     'New Brunswick',
     'Newfoundland and Labrador',
     'Nova Scotia',
     'Ontario',
     'Prince Edward Island',
     'Quebec',
     'Saskatchewan']




```python
from stats_can import StatsCan
import zipfile
import pandas as pd
#sc =StatsCan()

#job_vacancy=sc.table_to_df('14-10-0443-01')
```

    C:\Users\user\anaconda3\Lib\site-packages\stats_can\api_class.py:24: FutureWarning:
    
    This class will be deprecated in upcoming v3 release. Please see the docs for details
    
    C:\Users\user\anaconda3\Lib\site-packages\stats_can\sc.py:608: FutureWarning:
    
    This function will be deprecated in the v3 release. Please see the docs for details.
    
    C:\Users\user\anaconda3\Lib\site-packages\stats_can\sc.py:326: FutureWarning:
    
    This function will be deprecated in the v3 release. Please see the docs for details.
    
    C:\Users\user\anaconda3\Lib\site-packages\stats_can\sc.py:276: FutureWarning:
    
    This function will be deprecated in the v3 release. Please see the docs for details.
    
    

    Downloading and loading table_14100443
    

    14100443-eng.zip: 100%|██████████| 629M/629M [06:45<00:00, 1.55MB/s]  
    


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    File ~\anaconda3\Lib\site-packages\stats_can\sc.py:335, in table_from_h5(table, h5file, path)
        334     with pd.HDFStore(h5, "r") as store:
    --> 335         df = pd.read_hdf(store, key=table)
        336 except (KeyError, OSError):
    

    File ~\anaconda3\Lib\site-packages\pandas\io\pytables.py:451, in read_hdf(path_or_buf, key, mode, errors, where, start, stop, columns, iterator, chunksize, **kwargs)
        450         key = candidate_only_group._v_pathname
    --> 451     return store.select(
        452         key,
        453         where=where,
        454         start=start,
        455         stop=stop,
        456         columns=columns,
        457         iterator=iterator,
        458         chunksize=chunksize,
        459         auto_close=auto_close,
        460     )
        461 except (ValueError, TypeError, LookupError):
    

    File ~\anaconda3\Lib\site-packages\pandas\io\pytables.py:880, in HDFStore.select(self, key, where, start, stop, columns, iterator, chunksize, auto_close)
        879 if group is None:
    --> 880     raise KeyError(f"No object named {key} in the file")
        882 # create the storer and axes
    

    KeyError: 'No object named table_14100443 in the file'

    
    During handling of the above exception, another exception occurred:
    

    MemoryError                               Traceback (most recent call last)

    Cell In[510], line 6
          3 import pandas as pd
          4 sc =StatsCan()
    ----> 6 job_vacancy=sc.table_to_df('14-10-0443-01')
          8 zip_f=zipfile.ZipFile('/content/14100443-eng.zip')
          9 zip_f.extract(zip_f.infolist()[0])
    

    File ~\anaconda3\Lib\site-packages\stats_can\api_class.py:66, in StatsCan.table_to_df(self, table)
         48 def table_to_df(self, table):
         49     """Read a table to a dataframe.
         50 
         51     Parameters
       (...)
         64     call StatsCan.update_tables(), optionally passing just the table number of interest
         65     """
    ---> 66     return sc.table_to_df(table=table, path=self.data_folder, h5file="stats_can.h5")
    

    File ~\anaconda3\Lib\site-packages\stats_can\sc.py:613, in table_to_df(table, path, h5file)
        608 warn(
        609     "This function will be deprecated in the v3 release. Please see the docs for details.",
        610     FutureWarning,
        611 )
        612 if h5file:
    --> 613     df = table_from_h5(table=table, h5file=h5file, path=path)
        614 else:
        615     df = zip_table_to_dataframe(table=table, path=path)
    

    File ~\anaconda3\Lib\site-packages\stats_can\sc.py:338, in table_from_h5(table, h5file, path)
        336 except (KeyError, OSError):
        337     print("Downloading and loading " + table)
    --> 338     tables_to_h5(tables=table, h5file=h5file, path=path)
        339     with pd.HDFStore(h5, "r") as store:
        340         df = pd.read_hdf(store, key=table)
    

    File ~\anaconda3\Lib\site-packages\stats_can\sc.py:295, in tables_to_h5(tables, h5file, path)
        293 if not json_file.is_file():
        294     download_tables([table], path)
    --> 295 df = zip_table_to_dataframe(table, path=path)
        296 with open(json_file) as f_name:
        297     df_json = json.load(f_name)
    

    File ~\anaconda3\Lib\site-packages\stats_can\sc.py:195, in zip_table_to_dataframe(table, path)
        193         types_dict = {"VALUE": float}
        194         types_dict.update({col: str for col in col_names if col not in types_dict})
    --> 195         df = pd.read_csv(myfile, dtype=types_dict)
        197 possible_cats = [
        198     "GEO",
        199     "DGUID",
       (...)
        220     "Educational attainment",
        221 ]
        222 actual_cats = [col for col in possible_cats if col in col_names]
    

    File ~\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:948, in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
        935 kwds_defaults = _refine_defaults_read(
        936     dialect,
        937     delimiter,
       (...)
        944     dtype_backend=dtype_backend,
        945 )
        946 kwds.update(kwds_defaults)
    --> 948 return _read(filepath_or_buffer, kwds)
    

    File ~\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:617, in _read(filepath_or_buffer, kwds)
        614     return parser
        616 with parser:
    --> 617     return parser.read(nrows)
    

    File ~\anaconda3\Lib\site-packages\pandas\io\parsers\readers.py:1765, in TextFileReader.read(self, nrows)
       1762     else:
       1763         new_rows = len(index)
    -> 1765     df = DataFrame(col_dict, columns=columns, index=index)
       1767     self._currow += new_rows
       1768 return df
    

    File ~\anaconda3\Lib\site-packages\pandas\core\frame.py:733, in DataFrame.__init__(self, data, index, columns, dtype, copy)
        727     mgr = self._init_mgr(
        728         data, axes={"index": index, "columns": columns}, dtype=dtype, copy=copy
        729     )
        731 elif isinstance(data, dict):
        732     # GH#38939 de facto copy defaults to False only in non-dict cases
    --> 733     mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
        734 elif isinstance(data, ma.MaskedArray):
        735     from numpy.ma import mrecords
    

    File ~\anaconda3\Lib\site-packages\pandas\core\internals\construction.py:503, in dict_to_mgr(data, index, columns, dtype, typ, copy)
        499     else:
        500         # dtype check to exclude e.g. range objects, scalars
        501         arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    --> 503 return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    

    File ~\anaconda3\Lib\site-packages\pandas\core\internals\construction.py:152, in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
        149 axes = [columns, index]
        151 if typ == "block":
    --> 152     return create_block_manager_from_column_arrays(
        153         arrays, axes, consolidate=consolidate, refs=refs
        154     )
        155 elif typ == "array":
        156     return ArrayManager(arrays, [index, columns])
    

    File ~\anaconda3\Lib\site-packages\pandas\core\internals\managers.py:2091, in create_block_manager_from_column_arrays(arrays, axes, consolidate, refs)
       2089     raise_construction_error(len(arrays), arrays[0].shape, axes, e)
       2090 if consolidate:
    -> 2091     mgr._consolidate_inplace()
       2092 return mgr
    

    File ~\anaconda3\Lib\site-packages\pandas\core\internals\managers.py:1750, in BlockManager._consolidate_inplace(self)
       1744 def _consolidate_inplace(self) -> None:
       1745     # In general, _consolidate_inplace should only be called via
       1746     #  DataFrame._consolidate_inplace, otherwise we will fail to invalidate
       1747     #  the DataFrame's _item_cache. The exception is for newly-created
       1748     #  BlockManager objects not yet attached to a DataFrame.
       1749     if not self.is_consolidated():
    -> 1750         self.blocks = _consolidate(self.blocks)
       1751         self._is_consolidated = True
       1752         self._known_consolidated = True
    

    File ~\anaconda3\Lib\site-packages\pandas\core\internals\managers.py:2217, in _consolidate(blocks)
       2215 new_blocks: list[Block] = []
       2216 for (_can_consolidate, dtype), group_blocks in grouper:
    -> 2217     merged_blocks, _ = _merge_blocks(
       2218         list(group_blocks), dtype=dtype, can_consolidate=_can_consolidate
       2219     )
       2220     new_blocks = extend_blocks(merged_blocks, new_blocks)
       2221 return tuple(new_blocks)
    

    File ~\anaconda3\Lib\site-packages\pandas\core\internals\managers.py:2249, in _merge_blocks(blocks, dtype, can_consolidate)
       2246     new_values = bvals2[0]._concat_same_type(bvals2, axis=0)
       2248 argsort = np.argsort(new_mgr_locs)
    -> 2249 new_values = new_values[argsort]
       2250 new_mgr_locs = new_mgr_locs[argsort]
       2252 bp = BlockPlacement(new_mgr_locs)
    

    MemoryError: Unable to allocate 6.58 GiB for an array with shape (16, 55234368) and data type object



```python
zip_f=zipfile.ZipFile('14100443-eng.zip')
zip_f.extract(zip_f.infolist()[0])
```




    'C:\\Users\\user\\14100443.csv'




```python
nurses=['Registered nurses and registered psychiatric nurses','Nurse practitioners','Licensed practical nurses']
import pandas as pd
import pandas as pd

chunks=pd.read_csv('14100443.csv',iterator=True,chunksize=100000)
filtered_chunks=[]
for chunk in chunks:
  if 'National Occupational Classification' in chunk.columns:
    filtered_chunk= chunk[chunk['National Occupational Classification'].str.contains('|'.join(nurses),case=False,na=False)]
    filtered_chunks.append(filtered_chunk)
  else:
    print(chunk.columns)
    print('NOC not in columns')

job_vacancy=pd.concat(filtered_chunks,ignore_index= True)
```


```python
job_vacancy.columns
```




    Index(['REF_DATE', 'GEO', 'DGUID', 'National Occupational Classification',
           'Job vacancy characteristics', 'Statistics', 'UOM', 'UOM_ID',
           'SCALAR_FACTOR', 'SCALAR_ID', 'VECTOR', 'COORDINATE', 'VALUE', 'STATUS',
           'SYMBOL', 'TERMINATED', 'DECIMALS'],
          dtype='object')




```python
vacancy_data=job_vacancy[(job_vacancy['Job vacancy characteristics']=='Type of work, all types') &
(job_vacancy['Statistics']=='Job vacancies')]
```


```python
vacancy_data[vacancy_data.VALUE.notna()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REF_DATE</th>
      <th>GEO</th>
      <th>DGUID</th>
      <th>National Occupational Classification</th>
      <th>Job vacancy characteristics</th>
      <th>Statistics</th>
      <th>UOM</th>
      <th>UOM_ID</th>
      <th>SCALAR_FACTOR</th>
      <th>SCALAR_ID</th>
      <th>VECTOR</th>
      <th>COORDINATE</th>
      <th>VALUE</th>
      <th>STATUS</th>
      <th>SYMBOL</th>
      <th>TERMINATED</th>
      <th>DECIMALS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>266</th>
      <td>2015-01</td>
      <td>Canada</td>
      <td>2021A000011124</td>
      <td>Licensed practical nurses [32101]</td>
      <td>Type of work, all types</td>
      <td>Job vacancies</td>
      <td>Number</td>
      <td>223</td>
      <td>units</td>
      <td>0</td>
      <td>v1569992255</td>
      <td>1.462.1.1</td>
      <td>1685.0</td>
      <td>E</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5586</th>
      <td>2015-04</td>
      <td>Canada</td>
      <td>2021A000011124</td>
      <td>Registered nurses and registered psychiatric n...</td>
      <td>Type of work, all types</td>
      <td>Job vacancies</td>
      <td>Number</td>
      <td>223</td>
      <td>units</td>
      <td>0</td>
      <td>v1569991723</td>
      <td>1.458.1.1</td>
      <td>7060.0</td>
      <td>B</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5719</th>
      <td>2015-04</td>
      <td>Canada</td>
      <td>2021A000011124</td>
      <td>Nurse practitioners [31302]</td>
      <td>Type of work, all types</td>
      <td>Job vacancies</td>
      <td>Number</td>
      <td>223</td>
      <td>units</td>
      <td>0</td>
      <td>v1569991856</td>
      <td>1.459.1.1</td>
      <td>65.0</td>
      <td>D</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5852</th>
      <td>2015-04</td>
      <td>Canada</td>
      <td>2021A000011124</td>
      <td>Licensed practical nurses [32101]</td>
      <td>Type of work, all types</td>
      <td>Job vacancies</td>
      <td>Number</td>
      <td>223</td>
      <td>units</td>
      <td>0</td>
      <td>v1569992255</td>
      <td>1.462.1.1</td>
      <td>1685.0</td>
      <td>B</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5985</th>
      <td>2015-04</td>
      <td>Newfoundland and Labrador</td>
      <td>2021A000210</td>
      <td>Registered nurses and registered psychiatric n...</td>
      <td>Type of work, all types</td>
      <td>Job vacancies</td>
      <td>Number</td>
      <td>223</td>
      <td>units</td>
      <td>0</td>
      <td>v1570101315</td>
      <td>2.458.1.1</td>
      <td>135.0</td>
      <td>D</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>199367</th>
      <td>2024-04</td>
      <td>Alberta</td>
      <td>2021A000248</td>
      <td>Licensed practical nurses [32101]</td>
      <td>Type of work, all types</td>
      <td>Job vacancies</td>
      <td>Number</td>
      <td>223</td>
      <td>units</td>
      <td>0</td>
      <td>v1570978583</td>
      <td>10.462.1.1</td>
      <td>415.0</td>
      <td>E</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>199500</th>
      <td>2024-04</td>
      <td>British Columbia</td>
      <td>2021A000259</td>
      <td>Registered nurses and registered psychiatric n...</td>
      <td>Type of work, all types</td>
      <td>Job vacancies</td>
      <td>Number</td>
      <td>223</td>
      <td>units</td>
      <td>0</td>
      <td>v1571087643</td>
      <td>11.458.1.1</td>
      <td>3995.0</td>
      <td>D</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>199633</th>
      <td>2024-04</td>
      <td>British Columbia</td>
      <td>2021A000259</td>
      <td>Nurse practitioners [31302]</td>
      <td>Type of work, all types</td>
      <td>Job vacancies</td>
      <td>Number</td>
      <td>223</td>
      <td>units</td>
      <td>0</td>
      <td>v1571087776</td>
      <td>11.459.1.1</td>
      <td>110.0</td>
      <td>E</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>199766</th>
      <td>2024-04</td>
      <td>British Columbia</td>
      <td>2021A000259</td>
      <td>Licensed practical nurses [32101]</td>
      <td>Type of work, all types</td>
      <td>Job vacancies</td>
      <td>Number</td>
      <td>223</td>
      <td>units</td>
      <td>0</td>
      <td>v1571088175</td>
      <td>11.462.1.1</td>
      <td>1600.0</td>
      <td>C</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>200298</th>
      <td>2024-04</td>
      <td>Northwest Territories</td>
      <td>2021A000261</td>
      <td>Registered nurses and registered psychiatric n...</td>
      <td>Type of work, all types</td>
      <td>Job vacancies</td>
      <td>Number</td>
      <td>223</td>
      <td>units</td>
      <td>0</td>
      <td>v1571306827</td>
      <td>13.458.1.1</td>
      <td>70.0</td>
      <td>E</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 17 columns</p>
</div>




```python
vacancy_data['Year']=vacancy_data.REF_DATE.apply(lambda x: int(x.split('-')[0]))
can_vacancy_data=vacancy_data[['Year','GEO','National Occupational Classification','VALUE']]
can_vacancy_data=can_vacancy_data.groupby(['Year','National Occupational Classification','GEO'])['VALUE'].sum().reset_index()

```

    C:\Users\user\AppData\Local\Temp\ipykernel_4644\2149869845.py:1: SettingWithCopyWarning:
    
    
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    
    

****************************************************************************************************************************************************************************************************************************************************************************************************************


```python
sc=StatsCan()
can_population_all=sc.table_to_df('17-10-0009-01')
can_population_all['Year']=can_population_all.REF_DATE.apply(lambda x: x.year)
can_population=can_population_all[can_population_all.GEO == 'Canada']
can_population['Year']=can_population.REF_DATE.apply(lambda x: x.year)
can_pop_count=can_population[can_population.Year>=2014]

import pandas as pd
can_pop={'Year':[],'Population':[]}

for y in can_pop_count.Year.unique():
  can_pop['Year'].append(y)
  can_pop['Population'].append(can_pop_count[can_pop_count['Year']==y]['VALUE'].max())

can_pop=pd.DataFrame(can_pop)
can_pop
```

    C:\Users\user\anaconda3\Lib\site-packages\stats_can\api_class.py:24: FutureWarning: This class will be deprecated in upcoming v3 release. Please see the docs for details
      warn(
    C:\Users\user\anaconda3\Lib\site-packages\stats_can\sc.py:608: FutureWarning: This function will be deprecated in the v3 release. Please see the docs for details.
      warn(
    C:\Users\user\anaconda3\Lib\site-packages\stats_can\sc.py:326: FutureWarning: This function will be deprecated in the v3 release. Please see the docs for details.
      warn(
    C:\Users\user\AppData\Local\Temp\ipykernel_4644\275229224.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      can_population['Year']=can_population.REF_DATE.apply(lambda x: x.year)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014</td>
      <td>35555305.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>35823591.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016</td>
      <td>36257421.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>36722075.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018</td>
      <td>37259485.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019</td>
      <td>37828162.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2020</td>
      <td>38028638.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2021</td>
      <td>38446871.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2022</td>
      <td>39279501.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023</td>
      <td>40513781.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024</td>
      <td>41288599.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import requests

url="https://open.canada.ca/data/en/dataset/adad580f-76b0-4502-bd05-20c125de9116"
rec_id="adad580f-76b0-4502-bd05-20c125de9116"
base_url = "https://open.canada.ca/data/api/3/action"

def get_metadata(rec_id):
  url=f"{base_url}/package_show"
  params={'id':rec_id}
  options={'Accept':'app/json','Accept-Language':'en'}
  response=requests.get(url,params=params,headers=options)
  response.raise_for_status()
  return response.json()

def get_data(data_id):
  url=f"{base_url}/datastore_search"
  params={"resource_id":data_id,"limit":100000}
  response=requests.get(url,params=params)
  response.raise_for_status()
  return response.json()


```


```python
import numpy as np
import pandas as pd

meta=get_metadata(rec_id)
resources=meta['result']['resources']
salary_dict={}
#keys=['salary_2014','salary_2015','salary_2016','salary_2017','salary_2018','salary_2019','salary_2020','salary_2021','salary_2022','salary_2023']
years=np.arange(2014,2024).tolist()
years=[str(x) for x in years]

for i in resources:
    if i['name'].split()[0] in years:
      data_id=i['id']
      data=get_data(data_id)
      records=data['result']['records']
      salary_dict[i['name']]=pd.DataFrame(records)

noc_codes=['NOC_32101','NOC_31301','NOC_31302','NOC_3012','NOC_3233','NOC_3151','NOC_3152']
```


```python
for name,df in salary_dict.items():
  df.columns=df.columns.str.upper()
  df.columns=df.columns.str.replace(' ','_')
  df.columns=df.columns.str.replace('-','_')
  if 'NOC_CNP_2006' in df.columns:
    salary_dict[name]=df[df['NOC_CNP_2006'].isin(noc_codes)]
  else:
    salary_dict[name]=df[df['NOC_CNP'].isin(noc_codes)]
  if 'NOC_TITLE_ENG' in df.columns:
    df.rename(columns={'NOC_Title_ENG': 'NOC_Title'},inplace=True)

  elif 'NOC Title' in df.columns :
    df.rename(columns={'NOC Title': 'NOC_Title'},inplace=True)

  elif 'NOC_CNP_2006' in df.columns:
    df.rename(columns={'NOC_CNP_2006': 'NOC_CNP'},inplace=True)

  elif 'NOC_Title_E' in df.columns:
    df.rename(columns={'NOC_TITLE_ENG': 'NOC_Title'},inplace=True)
    salary_dict[name]=df

for name,df in salary_dict.items():
    for col in ['LOW_WAGE_SALAIRE_MINIUM','MEDIAN_WAGE_SALAIRE_MEDIAN', 'HIGH_WAGE_SALAIRE_MAXIMAL']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    if 'NOC_CNP_2006' in df.columns:
        new_df=df.groupby(['NOC_CNP_2006','PROV'])[['LOW_WAGE_SALAIRE_MINIUM','MEDIAN_WAGE_SALAIRE_MEDIAN', 'HIGH_WAGE_SALAIRE_MAXIMAL']].mean().reset_index()
    else:
        new_df=df.groupby(['NOC_CNP','PROV'])[['LOW_WAGE_SALAIRE_MINIUM','MEDIAN_WAGE_SALAIRE_MEDIAN', 'HIGH_WAGE_SALAIRE_MAXIMAL']].mean().reset_index()
    salary_dict[name]=new_df
#######################################################################################################################
Occupation_dict={'NOC_3151':'Head Nurses and Supervisors',
                 'NOC_32101':'Licensed Practical Nurse',
                 'NOC_3233':'Licensed Practical Nurse',
                 'NOC_31302':'Nurse Practitioners',
                 'NOC_31301':'Registered Nurse and Registered Psych. Nurse',
                 'NOC_3012':'Registered Nurse and Registered Psych. Nurse',
                 'NOC_3152':'Registered Nurse and Registered Psych. Nurse'}

for name,df in salary_dict.items():
  yr=int(name.split()[0])
  df['Year']=yr
  if 'NOC_CNP_2006' in df.columns:
    df.rename(columns={'NOC_CNP_2006': 'NOC_CNP'},inplace=True)

  df['Occupation']=df['NOC_CNP'].map(Occupation_dict)
  salary_dict[name]=df
```


```python
can_salary=pd.concat(salary_dict.values()).reset_index(drop=True)
```


```python
prov_dict={'AB':'Alberta','BC':'British Columbia','MB':'Manitoba','NA':'Canada','NB':'New Brunswick','NL':'Newfoundland and Labrador',
           'NS':'Nova Scotia','NT':'Northwest Territories','NU':'Nunavut','ON':'Ontario','PE':'Prince Edward Island','QC':'Québec',
           'SK':'Saskatchewan','YK':'Yukon','NW':'Northwest Territories','YT':'Yukon','CA':'Canada','National':'Canada'}
```


```python
for p in can_salary.PROV:
    if len(p) ==2:
        can_salary.PROV.replace(p,prov_dict[p],inplace=True)
    elif p=='National':
        can_salary.PROV.replace('National','Canada',inplace=True)
```


```python
cpi=sc.table_to_df('18-10-0005-01')
cpi
```

    C:\Users\user\anaconda3\Lib\site-packages\stats_can\sc.py:608: FutureWarning:
    
    This function will be deprecated in the v3 release. Please see the docs for details.
    
    C:\Users\user\anaconda3\Lib\site-packages\stats_can\sc.py:326: FutureWarning:
    
    This function will be deprecated in the v3 release. Please see the docs for details.
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REF_DATE</th>
      <th>GEO</th>
      <th>DGUID</th>
      <th>Products and product groups</th>
      <th>UOM</th>
      <th>UOM_ID</th>
      <th>SCALAR_FACTOR</th>
      <th>SCALAR_ID</th>
      <th>VECTOR</th>
      <th>COORDINATE</th>
      <th>VALUE</th>
      <th>STATUS</th>
      <th>SYMBOL</th>
      <th>TERMINATED</th>
      <th>DECIMALS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1914-01-01</td>
      <td>Canada</td>
      <td>2016A000011124</td>
      <td>All-items</td>
      <td>2002=100</td>
      <td>17</td>
      <td>units</td>
      <td>0</td>
      <td>v41693271</td>
      <td>2.2</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1914-01-01</td>
      <td>Canada</td>
      <td>2016A000011124</td>
      <td>All-items (1992=100)</td>
      <td>1992=100</td>
      <td>7</td>
      <td>units</td>
      <td>0</td>
      <td>v41713433</td>
      <td>2.309</td>
      <td>7.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>t</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1914-01-01</td>
      <td>Canada</td>
      <td>2016A000011124</td>
      <td>Goods and services</td>
      <td>2002=100</td>
      <td>17</td>
      <td>units</td>
      <td>0</td>
      <td>v41693519</td>
      <td>2.273</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>t</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1915-01-01</td>
      <td>Canada</td>
      <td>2016A000011124</td>
      <td>All-items</td>
      <td>2002=100</td>
      <td>17</td>
      <td>units</td>
      <td>0</td>
      <td>v41693271</td>
      <td>2.2</td>
      <td>6.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1915-01-01</td>
      <td>Canada</td>
      <td>2016A000011124</td>
      <td>All-items (1992=100)</td>
      <td>1992=100</td>
      <td>7</td>
      <td>units</td>
      <td>0</td>
      <td>v41713433</td>
      <td>2.309</td>
      <td>7.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>t</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>90195</th>
      <td>2023-01-01</td>
      <td>Yellowknife, Northwest Territories</td>
      <td>2011A00056106023</td>
      <td>Durable goods</td>
      <td>2002=100</td>
      <td>17</td>
      <td>units</td>
      <td>0</td>
      <td>v41695129</td>
      <td>30.275</td>
      <td>112.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90196</th>
      <td>2023-01-01</td>
      <td>Yellowknife, Northwest Territories</td>
      <td>2011A00056106023</td>
      <td>Semi-durable goods</td>
      <td>2002=100</td>
      <td>17</td>
      <td>units</td>
      <td>0</td>
      <td>v41695130</td>
      <td>30.276</td>
      <td>113.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90197</th>
      <td>2023-01-01</td>
      <td>Yellowknife, Northwest Territories</td>
      <td>2011A00056106023</td>
      <td>Non-durable goods</td>
      <td>2002=100</td>
      <td>17</td>
      <td>units</td>
      <td>0</td>
      <td>v41695131</td>
      <td>30.277</td>
      <td>190.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90198</th>
      <td>2023-01-01</td>
      <td>Yellowknife, Northwest Territories</td>
      <td>2011A00056106023</td>
      <td>Services</td>
      <td>2002=100</td>
      <td>17</td>
      <td>units</td>
      <td>0</td>
      <td>v41695132</td>
      <td>30.282</td>
      <td>155.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90199</th>
      <td>2023-01-01</td>
      <td>Iqaluit, Nunavut</td>
      <td>2011A00056204003</td>
      <td>All-items</td>
      <td>2002=100</td>
      <td>17</td>
      <td>units</td>
      <td>0</td>
      <td>v41713462</td>
      <td>31.2</td>
      <td>141.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>90200 rows × 15 columns</p>
</div>




```python
cpi_13=cpi[(cpi.UOM =='2002=100') & (cpi['Products and product groups']=='All-items')][['REF_DATE','GEO','Products and product groups','VALUE']]
cpi_13['Year']=cpi_13.REF_DATE.apply(lambda x: int(x.year))
cpi_13.drop('REF_DATE',axis=1,inplace=True)
cpi_13=cpi_13[cpi_13.Year >=2014]
cpi_13
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEO</th>
      <th>Products and product groups</th>
      <th>VALUE</th>
      <th>Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>69795</th>
      <td>Canada</td>
      <td>All-items</td>
      <td>125.2</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>70126</th>
      <td>Newfoundland and Labrador</td>
      <td>All-items</td>
      <td>128.4</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>70266</th>
      <td>St. John's, Newfoundland and Labrador</td>
      <td>All-items</td>
      <td>128.2</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>70272</th>
      <td>Prince Edward Island</td>
      <td>All-items</td>
      <td>130.1</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>70411</th>
      <td>Charlottetown and Summerside, Prince Edward Is...</td>
      <td>All-items</td>
      <td>129.3</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>89941</th>
      <td>Vancouver, British Columbia</td>
      <td>All-items</td>
      <td>154.1</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>89946</th>
      <td>Victoria, British Columbia</td>
      <td>All-items</td>
      <td>148.2</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>89951</th>
      <td>Whitehorse, Yukon</td>
      <td>All-items</td>
      <td>155.5</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>90075</th>
      <td>Yellowknife, Northwest Territories</td>
      <td>All-items</td>
      <td>156.9</td>
      <td>2023</td>
    </tr>
    <tr>
      <th>90199</th>
      <td>Iqaluit, Nunavut</td>
      <td>All-items</td>
      <td>141.9</td>
      <td>2023</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 4 columns</p>
</div>




```python
cpi_13['PROV']=[x.split(',')[1].strip() if ',' in x else x for x in cpi_13.GEO ]
cpi_14=cpi_13.groupby(['PROV','Year'])['VALUE'].mean().reset_index()
#cpi_14.PROV.replace('Yukon','Yukon Territory',inplace=True)
```


```python
rate_14=cpi_14[cpi_14.Year==2014][['PROV','VALUE']]
rate_14.rename(columns={'VALUE':'CPI_2014'},inplace=True)
cpi_14=cpi_14.merge(rate_14,on=['PROV'])
#cpi_14['REAL_CPI']=cpi_14['VALUE']/cpi_14['CPI_2014']
cpi_14.columns
```




    Index(['PROV', 'Year', 'VALUE', 'CPI_2014'], dtype='object')




```python
cpi_14['REAL_CPI']=cpi_14['VALUE']/cpi_14['CPI_2014']
cpi_14
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PROV</th>
      <th>Year</th>
      <th>VALUE</th>
      <th>CPI_2014</th>
      <th>REAL_CPI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alberta</td>
      <td>2014</td>
      <td>132.233333</td>
      <td>132.233333</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alberta</td>
      <td>2015</td>
      <td>133.800000</td>
      <td>132.233333</td>
      <td>1.011848</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alberta</td>
      <td>2016</td>
      <td>135.233333</td>
      <td>132.233333</td>
      <td>1.022687</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Alberta</td>
      <td>2017</td>
      <td>137.400000</td>
      <td>132.233333</td>
      <td>1.039072</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alberta</td>
      <td>2018</td>
      <td>140.833333</td>
      <td>132.233333</td>
      <td>1.065037</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>Yukon</td>
      <td>2019</td>
      <td>133.200000</td>
      <td>124.400000</td>
      <td>1.070740</td>
    </tr>
    <tr>
      <th>146</th>
      <td>Yukon</td>
      <td>2020</td>
      <td>134.500000</td>
      <td>124.400000</td>
      <td>1.081190</td>
    </tr>
    <tr>
      <th>147</th>
      <td>Yukon</td>
      <td>2021</td>
      <td>138.900000</td>
      <td>124.400000</td>
      <td>1.116559</td>
    </tr>
    <tr>
      <th>148</th>
      <td>Yukon</td>
      <td>2022</td>
      <td>148.300000</td>
      <td>124.400000</td>
      <td>1.192122</td>
    </tr>
    <tr>
      <th>149</th>
      <td>Yukon</td>
      <td>2023</td>
      <td>155.500000</td>
      <td>124.400000</td>
      <td>1.250000</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>




```python
can_adj_salary=pd.merge(can_salary,cpi_14, on=['Year','PROV'])
can_adj_salary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NOC_CNP</th>
      <th>PROV</th>
      <th>LOW_WAGE_SALAIRE_MINIUM</th>
      <th>MEDIAN_WAGE_SALAIRE_MEDIAN</th>
      <th>HIGH_WAGE_SALAIRE_MAXIMAL</th>
      <th>Year</th>
      <th>Occupation</th>
      <th>VALUE</th>
      <th>CPI_2014</th>
      <th>REAL_CPI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NOC_31301</td>
      <td>Alberta</td>
      <td>30.606667</td>
      <td>46.263333</td>
      <td>51.658889</td>
      <td>2023</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>164.400000</td>
      <td>132.233333</td>
      <td>1.243257</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NOC_31302</td>
      <td>Alberta</td>
      <td>36.105000</td>
      <td>53.293333</td>
      <td>65.925000</td>
      <td>2023</td>
      <td>Nurse Practitioners</td>
      <td>164.400000</td>
      <td>132.233333</td>
      <td>1.243257</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NOC_32101</td>
      <td>Alberta</td>
      <td>26.077778</td>
      <td>30.555556</td>
      <td>35.327778</td>
      <td>2023</td>
      <td>Licensed Practical Nurse</td>
      <td>164.400000</td>
      <td>132.233333</td>
      <td>1.243257</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NOC_31301</td>
      <td>British Columbia</td>
      <td>32.595556</td>
      <td>42.193333</td>
      <td>50.306667</td>
      <td>2023</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>151.166667</td>
      <td>118.900000</td>
      <td>1.271377</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NOC_31302</td>
      <td>British Columbia</td>
      <td>33.590000</td>
      <td>57.000000</td>
      <td>65.446667</td>
      <td>2023</td>
      <td>Nurse Practitioners</td>
      <td>151.166667</td>
      <td>118.900000</td>
      <td>1.271377</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>292</th>
      <td>NOC_3152</td>
      <td>Saskatchewan</td>
      <td>28.012857</td>
      <td>41.214286</td>
      <td>46.232857</td>
      <td>2014</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>129.000000</td>
      <td>129.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>293</th>
      <td>NOC_3233</td>
      <td>Saskatchewan</td>
      <td>21.253333</td>
      <td>32.550000</td>
      <td>34.980000</td>
      <td>2014</td>
      <td>Licensed Practical Nurse</td>
      <td>129.000000</td>
      <td>129.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>294</th>
      <td>NOC_3151</td>
      <td>Yukon</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014</td>
      <td>Head Nurses and Supervisors</td>
      <td>124.400000</td>
      <td>124.400000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>295</th>
      <td>NOC_3152</td>
      <td>Yukon</td>
      <td>34.360000</td>
      <td>41.638000</td>
      <td>73.713700</td>
      <td>2014</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>124.400000</td>
      <td>124.400000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>296</th>
      <td>NOC_3233</td>
      <td>Yukon</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014</td>
      <td>Licensed Practical Nurse</td>
      <td>124.400000</td>
      <td>124.400000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>297 rows × 10 columns</p>
</div>




```python
for col in ['LOW_WAGE_SALAIRE_MINIUM','MEDIAN_WAGE_SALAIRE_MEDIAN', 'HIGH_WAGE_SALAIRE_MAXIMAL']:
    name='REAL_'+col
    can_adj_salary[name]=can_adj_salary[col]*(can_adj_salary.REAL_CPI)

can_adj_salary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NOC_CNP</th>
      <th>PROV</th>
      <th>LOW_WAGE_SALAIRE_MINIUM</th>
      <th>MEDIAN_WAGE_SALAIRE_MEDIAN</th>
      <th>HIGH_WAGE_SALAIRE_MAXIMAL</th>
      <th>Year</th>
      <th>Occupation</th>
      <th>VALUE</th>
      <th>CPI_2014</th>
      <th>REAL_CPI</th>
      <th>REAL_LOW_WAGE_SALAIRE_MINIUM</th>
      <th>REAL_MEDIAN_WAGE_SALAIRE_MEDIAN</th>
      <th>REAL_HIGH_WAGE_SALAIRE_MAXIMAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NOC_31301</td>
      <td>Alberta</td>
      <td>30.606667</td>
      <td>46.263333</td>
      <td>51.658889</td>
      <td>2023</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>164.400000</td>
      <td>132.233333</td>
      <td>1.243257</td>
      <td>38.051949</td>
      <td>57.517207</td>
      <td>64.225268</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NOC_31302</td>
      <td>Alberta</td>
      <td>36.105000</td>
      <td>53.293333</td>
      <td>65.925000</td>
      <td>2023</td>
      <td>Nurse Practitioners</td>
      <td>164.400000</td>
      <td>132.233333</td>
      <td>1.243257</td>
      <td>44.887789</td>
      <td>66.257303</td>
      <td>81.961709</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NOC_32101</td>
      <td>Alberta</td>
      <td>26.077778</td>
      <td>30.555556</td>
      <td>35.327778</td>
      <td>2023</td>
      <td>Licensed Practical Nurse</td>
      <td>164.400000</td>
      <td>132.233333</td>
      <td>1.243257</td>
      <td>32.421376</td>
      <td>37.988404</td>
      <td>43.921502</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NOC_31301</td>
      <td>British Columbia</td>
      <td>32.595556</td>
      <td>42.193333</td>
      <td>50.306667</td>
      <td>2023</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>151.166667</td>
      <td>118.900000</td>
      <td>1.271377</td>
      <td>41.441224</td>
      <td>53.643613</td>
      <td>63.958714</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NOC_31302</td>
      <td>British Columbia</td>
      <td>33.590000</td>
      <td>57.000000</td>
      <td>65.446667</td>
      <td>2023</td>
      <td>Nurse Practitioners</td>
      <td>151.166667</td>
      <td>118.900000</td>
      <td>1.271377</td>
      <td>42.705537</td>
      <td>72.468461</td>
      <td>83.207354</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>292</th>
      <td>NOC_3152</td>
      <td>Saskatchewan</td>
      <td>28.012857</td>
      <td>41.214286</td>
      <td>46.232857</td>
      <td>2014</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>129.000000</td>
      <td>129.000000</td>
      <td>1.000000</td>
      <td>28.012857</td>
      <td>41.214286</td>
      <td>46.232857</td>
    </tr>
    <tr>
      <th>293</th>
      <td>NOC_3233</td>
      <td>Saskatchewan</td>
      <td>21.253333</td>
      <td>32.550000</td>
      <td>34.980000</td>
      <td>2014</td>
      <td>Licensed Practical Nurse</td>
      <td>129.000000</td>
      <td>129.000000</td>
      <td>1.000000</td>
      <td>21.253333</td>
      <td>32.550000</td>
      <td>34.980000</td>
    </tr>
    <tr>
      <th>294</th>
      <td>NOC_3151</td>
      <td>Yukon</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014</td>
      <td>Head Nurses and Supervisors</td>
      <td>124.400000</td>
      <td>124.400000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>295</th>
      <td>NOC_3152</td>
      <td>Yukon</td>
      <td>34.360000</td>
      <td>41.638000</td>
      <td>73.713700</td>
      <td>2014</td>
      <td>Registered Nurse and Registered Psych. Nurse</td>
      <td>124.400000</td>
      <td>124.400000</td>
      <td>1.000000</td>
      <td>34.360000</td>
      <td>41.638000</td>
      <td>73.713700</td>
    </tr>
    <tr>
      <th>296</th>
      <td>NOC_3233</td>
      <td>Yukon</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2014</td>
      <td>Licensed Practical Nurse</td>
      <td>124.400000</td>
      <td>124.400000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>297 rows × 13 columns</p>
</div>




```python
can_adj_salary[['LOW_WAGE_SALAIRE_MINIUM','MEDIAN_WAGE_SALAIRE_MEDIAN', 'HIGH_WAGE_SALAIRE_MAXIMAL',
                'REAL_LOW_WAGE_SALAIRE_MINIUM','REAL_MEDIAN_WAGE_SALAIRE_MEDIAN', 'REAL_HIGH_WAGE_SALAIRE_MAXIMAL']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LOW_WAGE_SALAIRE_MINIUM</th>
      <th>MEDIAN_WAGE_SALAIRE_MEDIAN</th>
      <th>HIGH_WAGE_SALAIRE_MAXIMAL</th>
      <th>REAL_LOW_WAGE_SALAIRE_MINIUM</th>
      <th>REAL_MEDIAN_WAGE_SALAIRE_MEDIAN</th>
      <th>REAL_HIGH_WAGE_SALAIRE_MAXIMAL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30.606667</td>
      <td>46.263333</td>
      <td>51.658889</td>
      <td>38.051949</td>
      <td>57.517207</td>
      <td>64.225268</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36.105000</td>
      <td>53.293333</td>
      <td>65.925000</td>
      <td>44.887789</td>
      <td>66.257303</td>
      <td>81.961709</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.077778</td>
      <td>30.555556</td>
      <td>35.327778</td>
      <td>32.421376</td>
      <td>37.988404</td>
      <td>43.921502</td>
    </tr>
    <tr>
      <th>3</th>
      <td>32.595556</td>
      <td>42.193333</td>
      <td>50.306667</td>
      <td>41.441224</td>
      <td>53.643613</td>
      <td>63.958714</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33.590000</td>
      <td>57.000000</td>
      <td>65.446667</td>
      <td>42.705537</td>
      <td>72.468461</td>
      <td>83.207354</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>292</th>
      <td>28.012857</td>
      <td>41.214286</td>
      <td>46.232857</td>
      <td>28.012857</td>
      <td>41.214286</td>
      <td>46.232857</td>
    </tr>
    <tr>
      <th>293</th>
      <td>21.253333</td>
      <td>32.550000</td>
      <td>34.980000</td>
      <td>21.253333</td>
      <td>32.550000</td>
      <td>34.980000</td>
    </tr>
    <tr>
      <th>294</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>295</th>
      <td>34.360000</td>
      <td>41.638000</td>
      <td>73.713700</td>
      <td>34.360000</td>
      <td>41.638000</td>
      <td>73.713700</td>
    </tr>
    <tr>
      <th>296</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>297 rows × 6 columns</p>
</div>




```python
test=can_adj_salary[can_adj_salary.PROV=='Canada'].groupby(['Year'])[['LOW_WAGE_SALAIRE_MINIUM','MEDIAN_WAGE_SALAIRE_MEDIAN', 'HIGH_WAGE_SALAIRE_MAXIMAL',
                'REAL_LOW_WAGE_SALAIRE_MINIUM','REAL_MEDIAN_WAGE_SALAIRE_MEDIAN', 'REAL_HIGH_WAGE_SALAIRE_MAXIMAL']].mean().reset_index()
```

## App Layout


```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import Dash, dcc, html, Input, Output,callback

new_app=Dash()

new_app.layout=html.Div(children=[
    html.Div([html.H1('Nursing Dashboard 2014-23',style={'text-align':'center','font':'sans-serif'})]),
    html.Div([
        dcc.Dropdown(can_population_all.GEO.cat.categories.tolist(),'Canada',id='selected_prov',style={"width": "400px", "font-size": "16px"})],
        style={"display": "flex", "justify-content": "center", "margin-bottom": "30px",'width':"400px"}),
    html.Div(style={"display": "flex", "justify-content": "space-between", "margin-bottom": "30px","flex-wrap":"flex"},
            children=[
             dcc.Graph(id='work_pop_graph',style={"flex": "6", "margin": "0 10px"}),
             dcc.Graph(id='workforce_pie',style={"flex": "4", "margin": "0 10px"})]),
    html.Div(style= {"display": "block", "margin-top": "20px"},
             children=[
                 dcc.Graph(id='salary_graph',style={"width": "100%", "margin-bottom": "20px"}),
                 dcc.Graph(id='vacancy_graph',style={"width": "100%"})])
        ])

@new_app.callback(
    Output('work_pop_graph','figure'),
    Input('selected_prov','value')
)
def update_workforce(selected_prov):
  #can_salary=can_salary[can_salary.PROV==selected_prov]
  supply_professionals['Jurisdiction']=supply_professionals['Jurisdiction'].str.replace('Provinces/territories with available data','Canada')
  workforce_professionals['Jurisdiction']=workforce_professionals['Jurisdiction'].str.replace('Provinces/territories with available data','Canada')
  supply_prof=supply_professionals[supply_professionals.Jurisdiction == selected_prov]
  workforce_prof=workforce_professionals[workforce_professionals.Jurisdiction==selected_prov]
  can_pop=can_population_all[(can_population_all.GEO==selected_prov) &
                             (can_population_all.Year >=supply_prof.Year.min())]
  can_pop=can_pop.groupby(['Year'])['VALUE'].max().reset_index()
  supply_prof=supply_prof.groupby(['Year'])['Supply: \nnumber \nof nurses'].sum().reset_index()
  workforce_prof=workforce_prof.groupby(['Year'])['Workforce: number \nof nurses'].sum().reset_index()

  fig1=make_subplots(specs=[[{"secondary_y":True}]])
  x1=supply_prof.Year
  y1=supply_prof['Supply: \nnumber \nof nurses']
  y2=workforce_prof['Workforce: number \nof nurses']
  y3=can_pop.VALUE

  fig1.add_trace(go.Bar(x=x1,y=y1,opacity=0.7,name='Supply',marker_color='navy'),secondary_y=False)
  fig1.add_trace(go.Bar(x=x1,y=y2,opacity=0.7,name='Workforce',marker_color='teal'),secondary_y=False)
  fig1.add_trace(go.Scatter(x=x1,y=y3,mode='lines+markers',name='Population',marker_color='crimson'),secondary_y=True)
  fig1.update_layout(title_text=f'Supply & Workforce vs. Population in {selected_prov}')
  fig1.update_yaxes(title_text='Supply/Workforce of professional staff',secondary_y=False)
  fig1.update_yaxes(title_text='Population',secondary_y=True)
  fig1.update_xaxes(title_text='Year')
  return fig1

@new_app.callback(
    Output('workforce_pie','figure'),
    Input('selected_prov','value')
)
def update_pie(selected_prov):
    workforce_professionals['Jurisdiction']=workforce_professionals['Jurisdiction'].str.replace('Provinces/territories with available data','Canada')
    workforce_prof=workforce_professionals[workforce_professionals.Jurisdiction==selected_prov]
    pie_data=workforce_prof.groupby(['Year','Type of professional'])['Workforce: number \nof nurses'].sum().reset_index()
    fig=go.Figure(data=[go.Pie(labels=pie_data['Type of professional'], values=pie_data['Workforce: number \nof nurses'], hole=.3)])
    return(fig)




@new_app.callback(
    Output('salary_graph','figure'),
    Input('selected_prov','value')
)
def update_salary(selected_prov):
    work_df=work_hrs[work_hrs.GEO==selected_prov]
    adj_sal=can_adj_salary[can_adj_salary.PROV==selected_prov].groupby(['Year'])[['MEDIAN_WAGE_SALAIRE_MEDIAN','REAL_MEDIAN_WAGE_SALAIRE_MEDIAN']].mean().reset_index()
    
    fig1=make_subplots(specs=[[{"secondary_y":True}]])
    x1=adj_sal.Year
    y1=work_df['VALUE']
    y2=adj_sal['MEDIAN_WAGE_SALAIRE_MEDIAN']
    y3=adj_sal['REAL_MEDIAN_WAGE_SALAIRE_MEDIAN']
    
    fig1.add_trace(go.Scatter(x=x1,y=y1,mode='lines+markers',name='Worked Hours',marker_color='navy'),secondary_y=True)
    fig1.add_trace(go.Bar(x=x1,y=y2,opacity=0.7,name='Nominal Median Wage',marker_color='teal'),secondary_y=False)
    fig1.add_trace(go.Bar(x=x1,y=y3,opacity=0.7,name='Real Median Wage',marker_color='crimson'),secondary_y=False)
    fig1.update_layout(title_text=f'Worked hours vs Real/Nominal Wages in {selected_prov}')
    fig1.update_yaxes(title_text='Real/Nominal Hourly Wages',secondary_y=False)
    fig1.update_yaxes(title_text='Worked Hours Per Person',secondary_y=True)
    fig1.update_xaxes(title_text='Year')
    return fig1

@new_app.callback(
    Output('vacancy_graph','figure'),
    Input('selected_prov','value')
)

def update_vacany(selected_prov):
    vacancy=can_vacancy_data[can_vacancy_data.GEO==selected_prov]
    #vacancy['Value'].replace(0.0,np.nan)
    fig=px.area(vacancy,x='Year',y='VALUE',color='National Occupational Classification',title=f'Vacancies in {selected_prov} by Occupation')
    fig.update_yaxes(title_text='Vacancies')
    return(fig)
new_app.run_server(jupyter_mode="external",debug=True)
```

    Dash app running on http://127.0.0.1:8050/
    


```python

```

{% endraw %}
