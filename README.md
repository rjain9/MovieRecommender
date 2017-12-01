# ALDA-Fall-2017

Following libraries are required to run:
- pandas
- numpy
- surprise
- operator

To run the project, following command can be used:
python aldaProjectMain.py "Justice League"

Sample Output:
'''
For movie (Justice League)
- Top 10 movies recommended based on KNN - 
            Year                            Name
Movie_Id                                        
48        2001.0                  Justice League
33        2000.0  Aqua Teen Hunger Force: Vol. 1
26        2004.0                 Never Die Alone
76        1952.0           I Love Lucy: Season 2
57        1995.0                     Richard III
17        2005.0                       7 Seconds
78        1996.0              Jingle All the Way
52        2002.0         The Weather Underground
79        1956.0                     The Killing
3         1997.0                       Character
For movie (Justice League)
- Top 10 movies recommended based on Pearsons'R correlation - 
PearsonR                            Name  count      mean
                                                         
1.000000                  Justice League   3591  3.710944
0.469184                 Never Die Alone   5861  2.793721
0.398932                     Richard III   3562  3.678551
0.345846         The Weather Underground   5147  3.757140
0.340054                  Spitfire Grill   8501  3.684037
0.329660                 Lilo and Stitch  39752  3.823254
0.314511                Immortal Beloved  10722  3.784369
0.311121           I Love Lucy: Season 2   2954  4.090386
0.299413  Rudolph the Red-Nosed Reindeer   6558  3.806496
0.283321                   Mostly Martha  11508  3.871828
'''


