###############
##  IMPORTS  ##
###############

from bs4 import BeautifulSoup
from functools import reduce
from pprint import pprint
import threading
import requests
import warnings
import random
import numpy
import time
import sys
import re
import os
warnings.filterwarnings("ignore")

#################
##  FUNCTIONS  ##
#################

# Logging Into indeed.com
def log_into_indeed(thread_id):
	# Sideline Number: 617-539-8135
	login_url = 'https://secure.indeed.com/account/login'
	credentials = {
		0:{'email':'samlatters@hotmail.com', 'password':'myp4ssword@1'},
		1:{'email':'jessicalim2@hotmail.com', 'password':'myp4ssword@2'},
		2:{'email':'sallymoss343@hotmail.com', 'password':'myp4ssword@3'},
		3:{'email':'lillydavis12@hotmail.com', 'password':'myp4ssword@4'},
		4:{'email':'robbietran@hotmail.com', 'password':'myp4ssword@5'},
		5:{'email':'tywatson899@hotmail.com', 'password':'myp4ssword@6'}
	}
	data = {
		'action':'Login',
		'__email':credentials[thread_id]['email'],
		'__password':credentials[thread_id]['password'],
		'remember':'1',
		'hl':'en',
		'continue':'/account/view?hl=en'
	}
	session_requests = requests.session()
	response = session_requests.post(login_url,data=data)
	return session_requests, data['__email']

# Pulling Webpage Content
def pull_webpage(url,session_requests):
	try:
		page = session_requests.get(url)
		soup = BeautifulSoup(page.content,'html.parser')
	except:
		time.sleep(30)
		page = session_requests.get(url)
		soup = BeautifulSoup(page.content,'html.parser')
	return soup

# Remove All Non-ASCII From A String
def clean_string(string):
	string = string.replace(u'\xa0\xa0','. ')
	string = re.sub('[.]{2,}','.',string)
	string = ''.join([i if ord(i) < 128 else ' ' for i in string])
	string = re.sub('\s+',' ',string).strip()
	return string

# Pulling Resumes From indeed.com
def indeed_search(thread_id,n_threads):

	# Logging Into Indeed
	session_requests, email = log_into_indeed(thread_id)

	resumes = {}
	resumes_count = {'thread_' + str(thread_id) : 0}
	f_out = open('resume_scraping_output_' + str(thread_id) + '.txt','w')
	wait_time = 4.0 + random.randint(0,2)

	# Iterating Over Job Types
	f_jobs = open('resume_scraping_job_titles.txt','r')
	job_types = []
	for line in f_jobs:
		job_type = line.replace('\n','')
		if len(job_type) > 2:
			job_types.append(job_type)
			resumes_count[job_type] = 0
	f_jobs.close()
	for job_type in job_types:

		# Grabbing Resume Links
		resumes[job_type] = {}
		resumes_offset = thread_id * 50
		url = 'https://www.indeed.com/resumes?q=' + job_type.replace(' ','+') + '&co=US&start=' + str(resumes_offset)
		time.sleep(wait_time)
		soup = pull_webpage(url,session_requests)
		resume_links = soup.find_all('a',class_='app_link')
		for resume_link in resume_links:
			href = resume_link['href']
			if '/r/' in href:
				resumes[job_type]['https://www.indeed.com' + href] = {}

		# Grabbing Resumes
		print 'thread: ' + str(thread_id) + ' (' + email + '); starting job type: ' + str(job_type) + '; num_resumes: ' + str(len(resumes[job_type]))
		if len(resumes[job_type]) == 0:
			time.sleep(300)
			continue
		for resume in resumes[job_type]:
			resumes[job_type][resume]['resume_link'] = resume
			soup = pull_webpage(resume,session_requests)

			# Basic Info
			basic_info = soup.find_all('div',id='basic_info_cell')
			if len(basic_info) > 0:
				# Resume Contact
				resume_contact = basic_info[0].find_all('h1',id='resume-contact')
				resume_contact = getattr(resume_contact[0],'text') if len(resume_contact) > 0 else ''
				# Resume Summary
				resume_summary = basic_info[0].find_all('p',id='res_summary')
				resume_summary = getattr(resume_summary[0],'text') if len(resume_summary) > 0 else ''
				resume_summary = clean_string(resume_summary) if resume_summary != '' else ''
				# Storing Basic Info
				resumes[job_type][resume]['resume_contact'] = resume_contact
				resumes[job_type][resume]['resume_summary'] = resume_summary
				# Checking Info Proper Info Exists
				if resumes[job_type][resume]['resume_contact'] == '':
					resumes[job_type][resume] = {}
					time.sleep(wait_time)
					continue
			else:
				resumes[job_type][resume] = {}
				time.sleep(wait_time)
				continue

			# Work Experience
			work_experience = soup.find_all('div',class_='section-item workExperience-content')
			if len(work_experience) > 0:
				for work_experience_elem in work_experience[0]:
					jobs = work_experience_elem.find_all('div',class_='work-experience-section')
					if len(jobs) > 0:
						# Iterating Over All Jobs
						job_ind = 0
						for job in jobs:
							# Job Name
							job_name = job.find_all('p',class_='work_title title')
							job_name = getattr(job_name[0],'text') if len(job_name) > 0 else ''
							# Job Company
							job_company = job.find_all('div',class_='work_company')
							job_company = getattr(job_company[0],'text') if len(job_company) > 0 else ''
							# Job Dates
							job_dates = job.find_all('p',class_='work_dates')
							job_dates = getattr(job_dates[0],'text') if len(job_dates) > 0 else ''
							# Job Description
							job_description = job.find_all('p',class_='work_description')
							job_description = getattr(job_description[0],'text') if len(job_description) > 0 else ''
							job_description = clean_string(job_description) if job_description != '' else ''
							# Storing Job Info
							job_idx = 'job_' + str(job_ind)
							resumes[job_type][resume][job_idx] = {}
							resumes[job_type][resume][job_idx]['job_name'] = job_name
							resumes[job_type][resume][job_idx]['job_company'] = job_company
							resumes[job_type][resume][job_idx]['job_dates'] = job_dates
							resumes[job_type][resume][job_idx]['job_description'] = job_description
							job_ind = job_ind + 1

			# Education
			education = soup.find_all('div',class_='section-item education-content')
			if len(education) > 0:
				for education_elem in education[0]:
					edu_programs = education_elem.find_all('div',class_='education-section')
					if len(edu_programs) > 0:
						# Iterating Over All Jobs
						edu_ind = 0
						for edu_program in edu_programs:
							# Education Title
							edu_title = edu_program.find_all('p',class_='edu_title')
							edu_title = getattr(edu_title[0],'text') if len(edu_title) > 0 else ''
							# Education School
							edu_school = edu_program.find_all('div',class_='edu_school')
							edu_school = getattr(edu_school[0],'text') if len(edu_school) > 0 else ''
							# Education Dates
							edu_dates = edu_program.find_all('p',class_='edu_dates')
							edu_dates = getattr(edu_dates[0],'text') if len(edu_dates) > 0 else ''
							# Storing Job Info
							edu_idx = 'edu_' + str(edu_ind)
							resumes[job_type][resume][edu_idx] = {}
							resumes[job_type][resume][edu_idx]['edu_title'] = edu_title
							resumes[job_type][resume][edu_idx]['edu_school'] = edu_school
							resumes[job_type][resume][edu_idx]['edu_dates'] = edu_dates
							edu_ind = edu_ind + 1

			# Skills
			skills = soup.find_all('div',class_='skill-container resume-element')
			skills = getattr(skills[0],'text') if len(skills) > 0 else ''
			resumes[job_type][resume]['skills'] = skills

			# Output And Tracking
			time.sleep(wait_time)
			cur_thread = 'thread_' + str(thread_id)
			resumes_count[cur_thread] = resumes_count[cur_thread] + 1
			resumes_count[job_type] = resumes_count[job_type] + 1
			f_out.write(str(resumes[job_type][resume]) + '\n\n')
			if resumes_count[cur_thread] % 25 == 0:
				print 'thread: ' + str(thread_id) + '; resume_count: ' + str(resumes_count[cur_thread]) + '; on job type: ' + str(job_type)

	f_out.close()
	return

# Threading Process For Pulling Resumes From indeed.com
def indeed_search_threading(n_threads=6):

	# Threading Object
	class resumeThread(threading.Thread):
		def __init__(self,threadID,n_threads):
			threading.Thread.__init__(self)
			self.threadID = threadID
		def run(self):
			indeed_search(self.threadID,n_threads)
			print "Exiting Thread " + str(self.threadID)

	# Running Threads
	thread_list = []
	for n in range(0,n_threads):
		thread_list.append(resumeThread(n,n_threads))
	for n in range(0,n_threads):
		thread_list[n].daemon=True
		thread_list[n].start()
	try:
		for thread in thread_list:
			while thread.isAlive():
				thread.join(1)
	except:
		print '\n! Received keyboard interrupt, quitting threads.\n'
		sys.exit()

	# Consolidating Output Files
	with open('resume_scraping_output.txt', 'w') as outfile:
		for n in range(0,n_threads):
			fname = 'resume_scraping_output_' + str(n) + '.txt'
			with open(fname) as infile:
				for line in infile:
					outfile.write(line)
			os.remove(fname)

############
##  MAIN  ##
############

def main():

	# Pulling Resumes From indeed.com
	indeed_search_threading()

if __name__ == '__main__':
	main()
