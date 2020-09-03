# TODO: cleanup


import pandas as pd
from bokeh.models import DataTable, ColumnDataSource, TableColumn, HTMLTemplateFormatter
from bokeh.plotting import save, output_file, figure


output_file('out/confusion_matrix.html')

data = pd.read_csv('out/confusion_matrix.csv')

data_arr = data.to_numpy()

data_dict = dict(enumerate(data_arr.tolist()))
data_dict = {str(d): data_dict[d] for d in data_dict.keys()}
data_dict['max'] = data_arr.max(axis=0)
data_dict['total'] = data_arr.sum(axis=0)

template = """
<div title=<%=(value/total*100).toPrecision(4)%>%>
<div style="position: absolute; background: #3477eb; height: calc(<%=value/total%>*100%); width: 100%; bottom: 0">
</div>
<div style="position: absolute; height: 100%; width: 100%; background:<%=
    (function colormax() {
        if (value == max) {
            return '#43eb34'
        }
    }())%>;
color: black">
<%=value %>
</div>
</div>
"""

formatter = HTMLTemplateFormatter(template=template)

columns = [TableColumn(field=i, title=i, formatter=formatter) for i in data.columns]

data_table = DataTable(source=ColumnDataSource(data_dict), columns=columns, sizing_mode="scale_both")

save(data_table)
