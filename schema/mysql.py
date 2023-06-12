import pymysql
 
conn = pymysql.connect(host = "project-db-stu.smhrd.com", user="seocho_0515_1", passwd="smhrd1", db="seocho_0515_1", port=3307, use_unicode=True, charset='utf8')
cursor = conn.cursor()