def Image_Scoring(reparsed_file, user_weights, startrow=0, limit=None):
    import pandas as pd
    import numpy as np
    import pickle
    import time
    import os
    from numpy.polynomial import Polynomial

    a = 7
    b = 2
    c = 1

    timenow = time.gmtime()
    year = timenow[0]
    month = timenow[1]
    day = timenow[2]
    print(f'The date is {year}_{month}_{day}')
    

    print('hi')
    starttime = time.time()



    
    
#Reading in pickle with the user weights and the reparsed file
    array_weights = pickle.load( open(user_weights, 'rb') )
    print('reading user weights pickle')

    #print(array_weights_old)

    #name_temp = 'Invincible.Fearless.Sensual.Mysterious.Enchanting.Vigorous.Diligent.Overwhelming.Gorgeous.Passionate.Terrifying.Beautiful.Powerful.Grey-Prince'

    #temp_truth = array_weights_old['user_name'] == name_temp[0:36]

    #print(array_weights_old[temp_truth])

    
    #array_weights = [n[0:36] for n in array_weights_old] 

    
    print(f'There are {len(array_weights)} users in the user weights array')
    #print(array_weights)
    #truth_Orion = array_weights['user_name'] == 'OrionNAU'
    #print(array_weights[truth_Orion])
    #temp = array_weights[truth_Orion]['7_2_1_func']
    # print(temp)
    # print(temp[0])
    # print(temp[0](10))
    # print('yuh')
    # print(yuh)

    # index = 0
    # for i in df_weights:
    #     print(i)
    #     index += 1
    #     if index == 5:
    #         continue
    # print(yuh)
    # print(df_weights[df_weights['user_name'] == 'OrionNAU'])
    # print(df_weights[df_weights['user_name'] == 'OrionNAU'][f'{a}_{b}_{c}_func'])

    #print(yuh)
    
    
    path = f'/scratch/wab77/official/{reparsed_file}'
    df = pd.read_csv(path, low_memory=False)
    print(len(df))

    batch_id = 0
    batch_length = len(np.unique(df['batch']))
    batch_counter = 0
    for batch_number in np.unique(df['batch']):
        print('batch number =', batch_number)
        # if batch_number != np.NaN:
        #     print('has a nan value surpassed my advanced security system?')
        if batch_counter  == batch_length - 1:
            print('batch counter = ', batch_counter, 'batch_length = ', batch_length)
            continue
        batch_id = batch_number
        batch_counter += 1
        
    print(batch_id, 'Is the batch_id actually an integer like it is supposed to be?')

    # print(len(np.unique(df['username'])))
    # print(len(np.unique(df['value'])))
    # print(len(np.unique(df['train'])))
    # print(df['thumbname'].nunique())





#constructing zeroes array for the image scores
    print('constructing zeros array')
    dtype_csv = [('thumb_name', 'U62'), ('score', float), ('batch_id', 'U40')]
    # print('hi')
    # print(len(np.unique(df['thumbname'])))
    test_images_analysis = np.zeros(df['thumbname'].nunique(), dtype=dtype_csv)


    
#Setting the thumbnames in the temp array equal to the names of the images we are itterating over
    print('setting names in array to names in df')
    test_images_analysis['thumb_name'] = df['thumbname'].unique() #12/11/24 np.unique will sort the thumbnames. Since they all start with numbers it puts them into numerical order, this is a problem if I just want to analyze the most recent batches. I am changing np.unique(df['thumbname']) to df['thumbname'].unique()

    print(len(test_images_analysis))

#iterating through all the images that have been classified
    #for i, image in enumerate(test_images['thumb_name']): 
        #d = df[df['thumbname'] == image] #dataframe for an image
        #denomenator = 0 #setting denomenator equal to 0
        #numerator = 0 #setting numerator equal to 0
        #for user in d['username']: #iterating through all the users for a particular image
            #truth_user = user_info_ult['user_name'] == user #truth aray for user in user_info_ult
            #if len(user_info_ult[truth_user]) == 1:
                #denomenator += (  (a * user_info_ult[truth_user]['tr_acc']) + (b * user_info_ult[truth_user]['classifications']) + (c * (1 - user_info_ult[truth_user]['ratio']))  ) #the denomenator gets added to 

                #if d[d['username'] == user].tail(1)['value'].values[0] == 'Yes': #if the user said yes to this image
                    #numerator += (  (a * user_info_ult[truth_user]['tr_acc']) + (b * user_info_ult[truth_user]['classifications']) + (c * (1 - user_info_ult[truth_user]['ratio']))  ) #numerator gets added to
        #try: #To avoid errors, I usually get the ZeroDivisionError because of an issue with the thumbnames
            #test_images_analysis['score'][i] = numerator / denomenator #Assigning the score for that image into the array
        #except ZeroDivisionError:
            #test_images_analysis['score'][i] = 0

#Creating a new directory to store the temporary information
    #print('making new directory')
    #outdir = f'/scratch/wab77/UMAA_v2_{a}_{b}_{c}'
    #os.makedirs(outdir, exist_ok=True)

#iterating through all the images that have been classified
    print('starting iteration through images')
    #starttime = time.time()
    marker = -1
    #marker1 = 0
    #marker2 = 0
    less_than_4_counter = 0
    
    for i, image in enumerate(test_images_analysis['thumb_name']):
        if i < startrow:
            continue
        if (limit != None) & (i >= startrow + limit):
            break
        marker += 1
        #marker1 += 1
        #print(i)
        print(image)
        if marker == 10:
            #marker1 += 1
            print(f'Starting on image {i} out of {len(test_images_analysis)}, still working ...')
            print(f'Time passedd: {time.time() - starttime}')
            print(numerator, denomenator)
            marker = 0
        d = df[df['thumbname'] == image] #dataframe for an image
        denomenator = 0
        numerator = 0 #starting off by setting numerator and denomenator equal to 0
        #print(len(d))

        if len(np.unique(d['value'])) == 1:
            image_value = np.unique(d['value'])
            if image_value == 'Yes':
                test_images_analysis['score'][i] = 1
            if image_value == 'No':
                test_images_analysis['score'][i] = 0
            test_images_analysis['batch_id'][i] = d['batch'].values[0]
            continue
            print('All user said the same thing for this image, continuing')


        
        #if d['batch'].values[0] == batch_id:
        for user in np.unique(d['username'][0:15]):#iterating over all users for a particular image
            print(user)
            print('Not all users said the same thing')
            # print(len(df_weights[df_weights['user_name' == user]]))
            # print(df_weights[df_weights['user_name' == user]][f'{a}_{b}_{c}_func'])
            # print(df_weights[df_weights['user_name' == user]][f'{a}_{b}_{c}_func'][0])
            #truth_user = user_info_ult_new_add_v2['user_name'] == user #truth array for a user in user_info_ult_new_add array
                                #truth_user1 = user_count['user_name'] == user
                                #truth_user2 = user_info['user_name'] == user
                                #if user_count['count'][truth_user1] < 10000:
                                    #user_count['count'][truth_user1] += 1
            e = df[df['username'] == user] #dataframe for a user
            f = e.to_numpy() #converting that dataframe to an array
            #If in the raparsed file there is a column for index then I need the extra '' as the ffirst column
            g = pd.DataFrame(f, columns = ['', 'username', 'started_at', 'finished_at', 'thumbname', 'train', 'value', 'retired', 'finished_workflow', 'selection_state', 'batch', 'subject_id', 'language', 'utc_offset', 'classification_id']) #converting that array back to a dataframe so I can index the row easily
            truth_user = array_weights['user_name'] == user#[0:36]
            poly = array_weights[truth_user][f'{a}_{b}_{c}_func'][0] #setting poly equal to the approximation for that user
            if len(d[d['username'] == user]) >= 1: #if the user saw that iamge once
                if len(e) < 4:
                    less_than_4_counter += 1
                if len(e) >= 4: #if the user has more than or equal to 4 classifications, below that the appoximations don't work well
                    index = g[g['thumbname'] == image].index[0] + 1 #setting the input value for poly (index) equal to the row number for that image for that user
                                        #print(user, index, poly(index))
                    if index > 10000:
                        index = 10000
                                        #index = user_count[truth_user1]['count']
                    denomenator += poly(index) #adding to the denomenator
                    if d[d['username'] == user].tail(1)['value'].values[0] == 'Yes': #if the user said yes adding to the unmerator
                        numerator += poly(index)
                                #if len(d[d['username'] == user]) > 1: #if the user saw this image more than once
                                    #if len(e) >= 4: #if they saw more than or equal to 4 images
                                        #for j, time in enumerate(d[d['username'] == user]['finished_at']): #iterating over every time a use saw that image, I am using the finished at time in orer to capture the order in whichh the user saw this image
                                            #index = g[g['finished_at'] == time].index[0] + 1 #Setting the input value for poly (index) equal to the time that user saw that image
                                            #denomenator += poly(index) #adding to denomenator
                                            #if d[d['username'] == user]['value'].values[j] == 'Yes': #if that user said yes that particualr time adding to numerator 
                                                #numerator += poly(index)
        #print(user, poly(index), numerator, denomenator)
        try: #Trying to record the score for each image, sometimes I get a dividde by zero error because something went wrong, usually with the image name, although i dont think this will be a real problem
            test_images_analysis['score'][i] = numerator / denomenator
        except ZeroDivisionError:
            test_images_analysis['score'][i] = 0

        test_images_analysis['batch_id'][i] = d['batch'].values[0]

        #if marker1 == 10000:
            #marker2 += 1
            #print('Saving temporary csv file for {marker1 * marker2} images')
            #dataframe_temp = pd.DataFrame(test_images_analysis)
            #dataframe_temp.to_csv(f'{outdir}/UMAA_v2_a{a}_b{b}_c{c}_{marker1 * marker2}.csv')
            #marker1 = 0


    print(f'the less than 4 counter was tripped {less_than_4_counter} times')
            
            
            
            
#truth array for all the images over the precentage that corresponds to that set of a,b,c
    #w = test_images_analysis['score'] >= percentage 
    
#number_of_images is equal to the amount of images above that cutoff percentage 
    #number_of_images = len(test_images_analysis[w]) 
    
#addding to the array the values we want to know
    #list_csv['a'] = a
    #list_csv['b'] = b
    #list_csv['c'] = c
    #list_csv['percentage'] = percentage
    #list_csv['number_of_images'] = number_of_images

#converting the array to a dataframe
    dataframe = pd.DataFrame(test_images_analysis) 
    
#writing the dataframe to a csv file
    outdir = f'/scratch/wab77/official/Results_Image_Scores/CitSci_Image_Analysis_{year}_{month}_{day}_{batch_id}'
    os.makedirs(outdir, exist_ok=True)
    dataframe.to_csv(f'{outdir}/CitSci_Image_Analysis_{year}_{month}_{day}_{batch_id}_{startrow}to{startrow + limit - 1}.csv')
    #dataframe.to_csv(f'/scratch/wab77/results_UMAA_v4_{a}_{b}_{c}.csv')

#writing the amount of time it took to iterate through all the images
    print('it took', time.time() - starttime, 'seconds? to complete the user weight calculations.')

###### SECTION: Command-line arguments configuration. ########
if __name__ == '__main__': # 4/9/19 COC: grouped all argparse items at end of file
    import argparse # This is to enable command line arguments.
    parser = argparse.ArgumentParser(description='Wills Ultimate Metric Analysis 2022/06/29')
#	parser.add_argument('--start-row', dest='start_row', type=int, default=0, help='starting row (slice from 0)')
#	parser.add_argument('file', metavar='O', help='CSV file of Hilda stuff from MPC scrape')
    #parser.add_argument('--a', dest='a', type=int, help='metric a')
    #parser.add_argument('--b', dest='b', type=int, help='metric b')
    #parser.add_argument('--c', dest='c', type=int, help='mwtric c')
    #parser.add_argument('--ucount', dest='ucount', type=int, help='ucount')
    parser.add_argument('--reparsed_file', dest='reparsed_file', type=str, help='reparsed_file')
    parser.add_argument('--user_weights', dest='user_weights', type=str, help='user_weights')
    parser.add_argument('--startrow', dest='startrow', type=int, help='start row', default=0)
    parser.add_argument('--limit', dest='limit', type=int, help='limit', default=None)
    

#	parser.add_argument('--observed-only', dest='observed', type=str, default='False', help='Only examine objects we actually have seen.') # default to False 10/22/19 COC: needed for ThumbNames but we will not yet know what we have (chicken/egg)
#	parser.add_argument('--fill-codes', dest='fill_codes', type=bool, help='Populate Codes Column', default=False) ## @todo fill-codes
#	parser.add_argument('--manual-items', dest='manual', type=bool, help='Update items we manually intervene with. 5/24/2020 COC', default=False)
#	parser.add_argument('--mode', dest='mode', type=str, default='default', help='Mode (default, distant-body so far)')
#	parser.add_argument('--specific-name', dest='specific_name', type=str, help='Specific object name to try') # 6/21/2021 COC

    args = parser.parse_args()
    print(f'Image_Scoring(reparsed_file={args.reparsed_file}), user_weights={args.user_weights}), startrow={args.startrow}, limit={args.limit}')
    Image_Scoring(reparsed_file=args.reparsed_file, user_weights=args.user_weights, startrow=args.startrow, limit=args.limit)
