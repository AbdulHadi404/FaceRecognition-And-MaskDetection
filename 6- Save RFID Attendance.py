import json
import csv
import urllib.request
import time
with urllib.request.urlopen("http://localhost/test/test.php") as url:
    data = json.loads(url.read().decode())

date = time.strftime('%Y-%m-%d')
fname = "rfidattendance\RFID_Attendance_Of_"+date+".csv"

with open(fname, "w") as file:
    csv_file = csv.writer(file,lineterminator='\n')
    csv_file.writerow(["id","username","serialnumber","card_uid","device_dep","checkindate","timein","timeout","card_out"])
    for item in data:
        csv_file.writerow([item['id'],item['username'],item['serialnumber'],item['card_uid'],item['device_dep'],item['checkindate'],item['timein'],item['timeout'],item['card_out']])
