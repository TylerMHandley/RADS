# RADS - Reccomendations using Anomoly Detection Systems
Just the first name that came to mind, can be changed later
## Inspiration
Modern day recommendation systems do not focus enough on exploring changing user tastes. To combat that we created this system to give extra emphasis to anomalies in a users listening habits and focus more on metrics such as novelty and serendipity instead of soley on accuracy.
## Related Works
Chang E.Y. (2011) PSVM: Parallelizing Support Vector Machines on Distributed Computers. In: Foundations of Large-Scale Multimedia Information Management and Retrieval. Springer, Berlin, Heidelberg
Yepes F.A., López V.F., Pérez-Marcos J., Gil A.B., Villarrubia G. (2018) Listen to This: Music Recommendation Based on One-Class Support Vector Machine. In: de Cos Juez F. et al. (eds) Hybrid Artificial Intelligent Systems. HAIS 2018. Lecture Notes in Computer Science, vol 10870. Springer, Cham
Zhang, Yuan & Séaghdha, Diarmuid & Quercia, Daniele & Jambor, Tamas. (2012). Auralist:     Introducing serendipity into music recommendation. WSDM 2012 - Proceedings of the 5th  ACM International Conference on Web Search and Data Mining. 13-22. 10.1145/2124295.2124300. 
## Dataset
### Million Song Subset
We are using a subset of the [Million Song Dataset](http://millionsongdataset.com/) that only contains 10,000 songs chosen at random from the entire dataset. However, the data is stored in the same format as the entire dataset meaning this approach can be generalized with no changes to the entire dataset. 
### Echo Nest Taste Profile Subset
The Million Song Dataset mostly provides the song and artist data, yet we still need user data. To this end we also use the [Echo Nest Taste Profile](http://millionsongdataset.com/tasteprofile/) in conjunction with the Million Song Dataset.
## Setup
The Taste Profile unlike the Million Song Subset is not broken into individual files. Therefore, to use the program you will need to download the zip from [here](http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip) and extract the contents of zip folder into the main directory. 
