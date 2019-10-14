Goto link: https://drive.google.com/file/d/1pvOFsIU8HUBcQNFgef5Y5QE54gPLU-YU/view?usp=sharing
to download the zip file mmehta63-HW2.zip

Prerequisites
	These instructions apply for Windows 10 x64. For testing on your own machine, you need to install the following libraries.

		ABAGAIL: https://github.com/pushkar/ABAGAIL
		Apache Ant: https://ant.apache.org/bindownload.cgi
		Java Development Kit: https://www.oracle.com/technetwork/java/javase/downloads/jdk10-downloads-4416644.html
		Add Java and Ant to your windows environment and path variables. A helpful guide is found at: https://www.mkyong.com/ant/how-to-install-apache-ant-on-windows/
	Once all of the prerequisites are installed, all of the methods are run from the Windows Command Prompt

Getting Started
	Download the dataset, PhishingWebsitesData_preprocessed.csv
	Edit the following .java files to point them towards your downloaded PhishingWebsitesData_preprocessed.csv file location. You can also use this time to edit the .java files to change the neurnal network structure
		phishing_rhc.java
		phishing_sa_val.java
		phishing_ga_val.java
		phishingwebsite_finaltest.java
	Run following from command prompt
		cd ABAGAIL
		BuildAndMove.bat
		runall.bat

Optimization_Results contains the results. Remove that folder from ABAGAIL before you run the 2 .bat files, so that a new set of results can be created and stored. And keep all folder structure intact upon extracting the zip file mmehta63-HW2.zip