import matplotlib.pyplot as plt
gdy = {
    'rouge-1':0.2596030939813189,
    'rouge-2':0.0975959698649965,
    'rouge-l':0.23041779564544812
}
nogdy = {
    'rouge-1':0.21608507699178328,
    'rouge-2':0.0729587204255833,
    'rouge-l':0.19027367296639522
}
nb5 = { 
    'rouge-1':0.27184995169177556, 
    'rouge-2':0.1096702155136363, 
    'rouge-l':0.2419045057194493 
    }
nb10 = { 
    'rouge-1':0.27282420728797874, 
    'rouge-2':0.11143031340681492, 
    'rouge-l':0.24224319922226237 
    }
tk10 = { 
    'rouge-1':0.23450846796492752, 
    'rouge-2':0.08244277365290231, 
    'rouge-l':0.2064400903042047 
    }
tk30 = { 
    'rouge-1':0.22323196494705166, 
    'rouge-2':0.0766941577870437, 
    'rouge-l':0.19680292384935466 
    }
tp69 = { 
    'rouge-1':0.22514035230348572, 
    'rouge-2':0.08095067883793959, 
    'rouge-l':0.19928168159205764
    }
tp87 = { 
    'rouge-1':0.20572181669786002, 
    'rouge-2':0.07039229106057791, 
    'rouge-l':0.18171030631906096
    }
tpr5 = {
    'rouge-1':0.25177119573283685,
    'rouge-2':0.09427045759547585,
    'rouge-l':0.22307245599127468
}
tpr7 = {
    'rouge-1':0.23553610995402086,
    'rouge-2':0.08410871388475848,
    'rouge-l':0.2092592065976719
}
values = [ i for i in range( 1, 11 )]
x = [
    'greedy', 
    'no_greedy',
    'num_beams : 5', 
    'num_beams : 10',
    'top_k : 10',
    'top_k : 30',
    'top_p : 0.69',
    'top_p : 0.87',
    'temperature : 0.5',
    'temperature : 0.7'
    ]
rouge_1 = [
    gdy['rouge-1'],
    nogdy['rouge-1'],
    nb5['rouge-1'],
    nb10['rouge-1'],
    tk10['rouge-1'],
    tk30['rouge-1'],
    tp69['rouge-1'],
    tp87['rouge-1'],
    tpr5['rouge-1'],
    tpr7['rouge-1']
]
rouge_2 = [
    gdy['rouge-2'],
    nogdy['rouge-2'],
    nb5['rouge-2'],
    nb10['rouge-2'],
    tk10['rouge-2'],
    tk30['rouge-2'],
    tp69['rouge-2'],
    tp87['rouge-2'],
    tpr5['rouge-2'],
    tpr7['rouge-2']
]
rouge_l = [
    gdy['rouge-l'],
    nogdy['rouge-l'],
    nb5['rouge-l'],
    nb10['rouge-l'],
    tk10['rouge-l'],
    tk30['rouge-l'],
    tp69['rouge-l'],
    tp87['rouge-l'],
    tpr5['rouge-l'],
    tpr7['rouge-l']
]
plt.bar( [i - 0.25 for i in values], rouge_1,label='rouge-1', width=0.25 )
plt.bar( values, rouge_2, label='rouge-2', width=0.25 )
plt.bar( [i + 0.25 for i in values], rouge_l,label='rouge-l', width=0.25 )
plt.legend() 
plt.ylabel("f1 score", size = 15 )
plt.xticks(values,x, rotation=45, ha='right', size = 10)
plt.show()