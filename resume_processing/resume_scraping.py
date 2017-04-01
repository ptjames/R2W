###############
##  IMPORTS  ##
###############

from bs4 import BeautifulSoup
from functools import reduce
from pprint import pprint
import requests
import warnings
import numpy
import sys
import re
warnings.filterwarnings("ignore")

#################
##  FUNCTIONS  ##
#################


# Pulling Resumes From bing.com
def bing_search():
    url = 'http://www.bing.com/images/search?q=resume%20examples'
    page = requests.get(url)
    soup = BeautifulSoup(page.content,'html.parser')
    images = [a['src'] for a in soup.find_all("img", {"src": re.compile("mm.bing.net")})]
    print images
    #print(soup.prettify())
    return


# Pulling Resumes From indeed.com
def indeed_search():
    '''
    Pulling resumes from indeed.com
    '''
    resumes = {}; resume_count = 0
    job_types = ['Analyst']
    for job_type in job_types:

        # Grabbing Resume Links
        resumes[job_type] = {}
        url = 'https://www.indeed.com/resumes?q=' + job_type
        page = requests.get(url)
        soup = BeautifulSoup(page.content,'html.parser')
        resume_links = soup.find_all('a',class_='app_link')
        for resume_link in resume_links:
            href = resume_link['href']
            if '/r/' in href:
                resumes[job_type]['https://www.indeed.com' + href] = {}

        # Grabbing Resumes
        for resume in resumes[job_type]:
            page = requests.get(resume)
            soup = BeautifulSoup(page.content,'html.parser')

            # Basic Info
            basic_info = str(soup.find_all('div',id='basic_info_cell'))
            if 'res_summary">' in basic_info:
                res_summary = basic_info.split('res_summary">')[1].split('</p>')[0].replace('\\xa0<br/>','\n')
                resumes[job_type][resume]['res_summary'] = res_summary
            else:
                resumes[job_type][resume]['res_summary'] = ''

            # Work Experience
            work_experience = soup.find_all('div',class_='section-item workExperience-content')
            for work_experience_elem in work_experience[0]:
                jobs = work_experience_elem.find_all('div',class_='work-experience-section')
                if len(jobs) > 0:
                    # Iterating Over All Jobs
                    job_ind = 0
                    for job in jobs:
                        job_name = getattr(job.find_all('p',class_='work_title title')[0],'text')
                        work_company = getattr(job.find_all('div',class_='work_company')[0],'text')
                        work_dates = getattr(job.find_all('p',class_='work_dates')[0],'text')
                        # Formatting Job Description
                        work_description = job.find_all('p',class_='work_description')[0]
                        work_description = str(work_description).decode('ascii',errors='ignore').split('work_description">')[1].split('<br/>')
                        work_description = reduce((lambda x, y: x + '\n' + y), [w.lstrip() for w in work_description]).replace('</p>','')
                        # Storing Job Info
                        job_idx = 'job_' + str(job_ind)
                        resumes[job_type][resume][job_idx] = {}
                        resumes[job_type][resume][job_idx]['job_name'] = job_name
                        resumes[job_type][resume][job_idx]['work_company'] = work_company
                        resumes[job_type][resume][job_idx]['work_dates'] = work_dates
                        #resumes[job_type][resume][job_idx]['work_description'] = work_description
                        resumes[job_type][resume][job_idx]['full_section'] = '\n'.join([job_name,work_company,work_dates,work_description])
                        job_ind = job_ind + 1

            # Education
            education = soup.find_all('div',class_='section-item education-content')
            for education_elem in education[0]:
                edu_programs = education_elem.find_all('div',class_='education-section')
                if len(edu_programs) > 0:
                    # Iterating Over All Jobs
                    edu_ind = 0
                    for edu_program in edu_programs:
                        edu_title = getattr(edu_program.find_all('p',class_='edu_title')[0],'text')
                        edu_school = getattr(edu_program.find_all('div',class_='edu_school')[0],'text')
                        edu_dates = getattr(edu_program.find_all('p',class_='edu_dates')[0],'text')
                        # Storing Job Info
                        edu_idx = 'edu_' + str(edu_ind)
                        resumes[job_type][resume][edu_idx] = {}
                        resumes[job_type][resume][edu_idx]['edu_title'] = edu_title
                        resumes[job_type][resume][edu_idx]['edu_school'] = edu_school
                        resumes[job_type][resume][edu_idx]['edu_dates'] = edu_dates
                        resumes[job_type][resume][edu_idx]['full_section'] = '\n'.join([edu_title,edu_school,edu_dates])
                        edu_ind = edu_ind + 1

            # Skills
            skills = soup.find_all('div',class_='section-item skills-content')
            #print skills
            #print ''

            resume_count = resume_count + 1
            if resume_count > 1:
                print resume
                break

    return resumes

############
##  MAIN  ##
############

resumes = indeed_search()
pprint(resumes)
