## 这是我根据自己的数据集改写的一版，这里存的是一些区别源代码的数据，以下是一些文件的解释：
## book_data.ttl/model_data.ttl 是原始数据SPO
## book_labelset.txt/model_labelset.txt 是上面数据源的SO本体类型。原作者构建了查询语句，此处可以效仿./for_prox_graph/KB_entity_type.py 编写cypher查询语句，返回SO的本体。由于时间原因，该功能相关暂未编写，敬请期待。
## labels_p_labelo.txt 融合上面的四个文件，它把两个数据源贴在一起，将S O替换成 S_label O_label
## match.txt 是经由LLM从book_labelset.txt/model_labelset.txt选出来的一些相似的label_pair
## examplebook_pred_prox_graph_matched.pickle 是由match_type_example.py 根据match.txt 对 labels_p_labelo.txt 进行label休整，获得谓词邻接图。参与./code/my_AutoAlign训练
## map.ttl是实体对齐验证集，注意实体前后关系，book在前，example在后
