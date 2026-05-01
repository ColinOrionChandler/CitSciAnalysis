def UserWeights(input_file):
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
    print(f'The date is {year}_{month}{day}')
    

    print('hi, this is Will, yuhhhhhhhhh')
    starttime = time.time()

    
#Loading in training image array to use as something to tell me if a trianing image is problematic
    #This cell will load in the array so I don't have to run the above cell again
    datafile6 = 'train_images_new_040424_sans_png.pkl' # name of the file to load, 040424 WAB changed the pickled array so the image names no longer end in .png and can be correcctly searched through to identify problematic training images
    train_images = pickle.load( open(datafile6, 'rb') )
    print('training image array has been loaded in')


    
#Opening entire data file
    print('reading reparsed file')
    path = f'/scratch/wab77/official/{input_file}'
    df = pd.read_csv(path, low_memory=False)
    print(len(df))

    batch_id = 0
    batch_length = len(np.unique(df['batch'])) #12/11/24 I found out that np.unique will sort everything, specifically, into numerical order I think it is fine here where I am printing out the most recent batch
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



    
#Creating array to hhold the users weights
    dtype = [('user_name', 'U150'), ('logged in Yes/No', 'U8'), ('train_count', int), ('test_count', int), 
           (f'{a}_{b}_{c}_func', 'object' ), ('arg_id', int)]
    user_info_ult_new_add_v2 = np.zeros(len(df['username'].unique()), dtype=dtype) #12/11/24 changing np.unique(df['username']) to df['username'].unique()
    user_info_ult_new_add_v2['user_name'] = df['username'].unique() #12/11/24 I am also changing no.unique to df.unique()
    #user_info_ult_new_mult_v2['logged in Yes/No'] = user_info['logged in Yes/No']
    print('Array has been created to hold user weights')

    
    counter = -1
#Calculating a users weight over time
    for i, user in enumerate(user_info_ult_new_add_v2['user_name']):
        counter += 1 #used to show me the progress of the cell
        #print(f'Starting on user {counter} out of {len(np.unique(df['username']))}, still working ..')
        print(f'Starting on user {counter} out of many, still working ...')
        
        #I am creating a bunch of empty lists to hold the different values of the weights over time
        #these lists are used to hold the weights over time for each user and used to calculate the approximation
        weight = []

        #dataframe for a particular user
        user_df = df[df['username'] == user] #user_df = df[df['username'].str[:36] == user]
        print(user)
        
        #list of multiples of 10 that will be used to calculate the approximation
        #only need one list of x values
        x = [] 
        
        #temp_list = [] #
        #truth_user = user_info['user_name'] == user #truth array for a user in user_info
        classifications = len(user_df)
        print(f'Classifications = {classifications}')
        if classifications > 10000:
            classifications = 10000
        
        test_yes_count = 0
        test_count = 0
        train_yes_count = 0
        train_count = 0
            
        counter1 = -1
        counter2 = -1
        
        #These lines are used in the case that a user has looked at an image multiple times
        #images list is the list of images for a user, when I loop through all the images I will loop through a sorted list
        #so if they have see the same image multiple times it will iterate through that image in succesion
        #if the image is the same as the previous one then temp_image_index is increased by one for each repetitiion
        #at the end of each loop image_temp will be updated to the image that was just looped through, that way I can test
        #to see if the same image was looked at multiple times and record the yes's and no's depending on which 
        #time they saw that image
        
        #imageslist = b['thumbname'].values[0:10000]
        #temp_image_index = -1
        #image_temp = 'none'
        
        
        #Creating an empty list to hold the images that a user looks at
        temp_image_list = []
        
        #This if statement will distinguish between users who have seen more than 10 images and users who have seen less than 10 images
        if classifications >= 100:
            counter3 = -1
            #iterating through their first 10,000 images, or all of their images if they have classified less, training and test
            for image in user_df['thumbname'].values[0:10000]: 
                counter3 += 1
                if counter3 == 1000:
                    #print(f'reached image {counter3} out of {classifications} for this user')
                    counter3 = -1
                counter1 += 1
                counter2 += 1
                if counter1 == 10: #after every 10 images
                    try:
                        z = ( a * (train_yes_count / train_count) + b * (np.log10(train_count + test_count) / 4) + c * (1 - (test_yes_count / test_count)) ) / (a + b + c)

                        #weight.append(z)
                        #x.append(counter2)
                    #If i get a divide by zero error the except will run these instead
                    except ZeroDivisionError as msg:
                    #print(f'yo {msg}, {user, train_count, test_count}')
                        #print(f'{user, train_count, test_count}')
                        if (test_count == 0) & (train_count == 0):
                            z = 0
                        elif (train_count == 0) & ((b + c) == 0):
                            z = 0
                        elif (test_count == 0) & ((a + b) == 0):
                            z = 0
                        elif (train_count != 0) & ((a + b) != 0):
                            #print((a + b), train_yes_count, train_count, test_count, test_yes_count)
                            z = ( a * (train_yes_count / train_count) + b * (np.log10(train_count + test_count) / 4) ) / (a + b)
                        elif (test_count != 0) & ((b + c) != 0):
                            z = ( b * (np.log10(train_count + test_count) / 4) + c * (1 - (test_yes_count / test_count)) ) / (b + c)
                            
                    #resetting the counter1 at the end of the if statement 
                    counter1 = 0
                    #appending to x at the end of the if statement
                    x.append(counter2)
                    #appending to the different weight lists
                    weight.append(z)
             

                #in this part of the for loop each image a user looks through is evaluated to see if that image is a train or 
                #tewst iamge, and whether the user said yes or no to that iamge
                #since we have a problem where people could have seen the same image multiple times I am using this index
                #trick so on the 5th iteration of a particular image I am checking the 5th answer that that user gave

                #dataframe for a user and iamge
                user_image_df = user_df[user_df['thumbname'] == image] 
                print('Classifications >= 100')

                #this if statement is important if a user lookedd at the same image twice, if they have then temp_image_index
                #will be increased by 1, if not then it will be set to 0. This lets be index the value that a user said
                #for a particular image, and on the particular time through seeing that image
    #             if image == image_temp:
    #                 temp_image_index += 1
    #             else:
    #                 temp_image_index = 0

                #These lines will append to a list all of the iamges a user has looked at, and temp_image_index is equal to the
                #number of times that image appears in taht list minus 1
                temp_image_list.append(image)
                temp_image_index = temp_image_list.count(image) - 1
                if len(user_image_df['train']) == 1:
                    print('Train value for this image =', user_image_df['train'].values[0])
                

                #If the image is not a training image, I have the greater than 0 condition cuz people can see the same image
                #multiple times
                if len(user_image_df['train']) == 1:
                    if user_image_df['train'].values[0] == False: 
                        print('This is a test/real image')
                        test_count += 1
                        if user_image_df['value'].values[temp_image_index] == 'Yes': #if user said yes to this test image
                            print('This user said yes')
                            test_yes_count += 1
                    else: #for training images
                        truth_train = train_images['thumb_name'] == image
                        if train_images[truth_train].size == 0:
                            print('Hi, this training image is not in the train images array') #Assuming its not problematic
                            train_count += 1
                            if user_image_df['value'].values[temp_image_index] == 'Yes': #if user said yes to training image
                                train_yes_count += 1
                        else:
                            print('This training image is in the train image array')
                            if train_images[truth_train]['Problematic Yes/No'] == 'No':
                                print('This training image is not problematic') #If it is problematic it gets skipped
                                train_count += 1
                                if user_image_df['value'].values[temp_image_index] == 'Yes': #if user said yes to training image
                                    train_yes_count += 1
                            else:
                                print('This training image is problematic, not using')
    
                    print(f'test count = {test_count}, test yes count = {test_yes_count}, train count = {train_count}, train yes count = {train_yes_count}')
                    #Setting the name of image_temp tp the last image that was iterated over        
                    #image_temp = image

                
        else: #if the users have less than 100 classified images, this else will run. In this case the append wills happen after every image not just every 10 images 
            for image in user_df['thumbname'].values: 
    #             counter3 += 1
    #             if counter3 == 1000:
    #                 print(f'reached image {counter3} out of {classifications} for this user')
    #                 counter3 = -1
    #             counter1 += 1
                counter2 += 1
                #if counter1 == 10: #after every 10 images
               

                #in this part of the for loop each image a user looks through is evaluated to see if that image is a train or 
                #tewst iamge, and whether the user said yes or no to that iamge
                #since we have a problem where people could have seen the same image multiple times I am using this index
                #trick so on the 5th iteration of a particular image I am checking the 5th answer that that user gave

                #dataframe for a user and iamge
                user_image_df = user_df[user_df['thumbname'] == image]
                print('Classifications < 100')

                #this if statement is important if a user lookedd at the same image twice, if they have then temp_image_index
                #will be increased by 1, if not then it will be set to 0. This lets be index the value that a user said
                #for a particular image, and on the particular time through seeing that image
    #             if image == image_temp:
    #                 temp_image_index += 1
    #             else:
    #                 temp_image_index = 0


                #These lines will append to a list all of the iamges a user has looked at, and temp_image_index is equal to the
                #number of times that image appears in taht list minues 1
                temp_image_list.append(image)
                temp_image_index = temp_image_list.count(image) - 1

            
                #If the image is not a training image, I have the greater than 0 condition cuz people can see the same image
                #multiple times

                #042124 WAB: I got an error with the line user_image_df['train'].values[0]. Indexx 0 is out of bounds for axis 0 of size 0. I am adding tihs if the length is 1 statement to try and circumvent that. 
                if len(user_image_df['train']) == 1:
                    if user_image_df['train'].values[0] == False: 
                        test_count += 1
                        if user_image_df['value'].values[temp_image_index] == 'Yes': #if user said yes to this test image
                            test_yes_count += 1
                    else: #for training images
                        truth_train = train_images['thumb_name'] == image
                        if train_images[truth_train].size == 0:
                            print('Hi, this training image is not in the train images array') #Assuming its not problematic
                            train_count += 1
                            if user_image_df['value'].values[temp_image_index] == 'Yes': #if user said yes to training image
                                train_yes_count += 1
                        else:
                            print('This training image is in the train image array')
                            if train_images[truth_train]['Problematic Yes/No'] == 'No':
                                print('This training image is not problematic') #It it is problematic it gets skipped
                                train_count += 1
                                if user_image_df['value'].values[temp_image_index] == 'Yes': #if user said yes to training image
                                    train_yes_count += 1
                            else:
                                print('This training image is problematic, not using')

                        
                try:
                    z = ( a * (train_yes_count / train_count) + b * (np.log10(train_count + test_count) / 4) + c * (1 - (test_yes_count / test_count)) ) / (a + b + c)


                    #weight.append(z)
                    #x.append(counter2)
                #If i get a divide by zero error the except will run these instead
                except ZeroDivisionError as msg:
                #except ZeroDivisionError as msg:
                    #print(f'yo {msg}, {user, train_count, test_count}')
                    #print(f'{user, train_count, test_count}')
                    if (test_count == 0) & (train_count == 0):
                        z = 0
                    elif (train_count == 0) & ((b + c) == 0):
                        z = 0
                    elif (test_count == 0) & ((a + b) == 0):
                        z = 0
                    elif (train_count != 0) & ((a + b) != 0):
                        #print((a + b), train_yes_count, train_count, test_count, test_yes_count)
                        z = ( a * (train_yes_count / train_count) + b * (np.log10(train_count + test_count) / 4) ) / (a + b)
                    elif (test_count != 0) & ((b + c) != 0):
                        z = ( b * (np.log10(train_count + test_count) / 4) + c * (1 - (test_yes_count / test_count)) ) / (b + c)
                        


                #resetting the counter1 at the end of the if statement 
                #counter1 = 0
                #appending to x at the end of the if statement
                x.append(counter2)
                #appending to the different weight lists
                weight.append(z)

                    
                #Setting the name of image_temp tp the last image that was iterated over        
                #image_temp = image
                
                

        #recording the different fits for a user
        #adding an if statement for users that have seen more than 1 image
        #print(user, classifications)
        if classifications >= 2:
            user_info_ult_new_add_v2[f'{a}_{b}_{c}_func'][i] = Polynomial.fit(x, weight, deg=5)

            
        else:
            try:
                user_info_ult_new_add_v2[f'{a}_{b}_{c}_func'][i] = weight[0]
            except IndexError as Ierr:
                user_info_ult_new_add_v2[f'{a}_{b}_{c}_func'][i] = 0
  
    


    #converting array to a dataframe 
    #dataframe = pd.DataFrame(user_info_ult_new_add_v2)
    #041724 When I read the csv file during the image scoring part the func does not come back as a polynomial function
    #I am going to pickle the file instead

    
#writing the dataframe to a csv file
    datafile = f'/scratch/wab77/official/UserWeights_{year}_{month}_{day}_Batch{batch_id}.pkl'
    #os.makedirs(outdir, exist_ok=True)
    print('saving data to', datafile)
    pickle.dump(user_info_ult_new_add_v2, open(datafile, 'wb'))
    #dataframe.to_csv(f'{outdir}/UserWeights_{year}_{month}_{day}_Batch{batch_id}.csv')
    #dataframe.to_csv(f'/scratch/wab77/results_UMAA_v4_{a}_{b}_{c}.csv')

#writing the amount of time it took to iterate through all the images
    print('it took', time.time() - starttime, 'seconds? to complete the user weight calculations.')

###### SECTION: Command-line arguments configuration. ########
if __name__ == '__main__': # 4/9/19 COC: grouped all argparse items at end of file
    import argparse # This is to enable command line arguments.
    parser = argparse.ArgumentParser(description='User Weight Calculation 2024/04/12')
#	parser.add_argument('--start-row', dest='start_row', type=int, default=0, help='starting row (slice from 0)')
#	parser.add_argument('file', metavar='O', help='CSV file of Hilda stuff from MPC scrape')
    #parser.add_argument('--a', dest='a', type=int, help='metric a')
    #parser.add_argument('--b', dest='b', type=int, help='metric b')
    #parser.add_argument('--c', dest='c', type=int, help='mwtric c')
    #parser.add_argument('--ucount', dest='ucount', type=int, help='ucount')
    #parser.add_argument('--limit', dest='limit', type=int, help='limit', default=None)
    #parser.add_argument('--startrow', dest='startrow', type=int, help='start row', default=0)
    parser.add_argument('--input_file', dest='input_file', type=str, help='input_file')

#	parser.add_argument('--observed-only', dest='observed', type=str, default='False', help='Only examine objects we actually have seen.') # default to False 10/22/19 COC: needed for ThumbNames but we will not yet know what we have (chicken/egg)
#	parser.add_argument('--fill-codes', dest='fill_codes', type=bool, help='Populate Codes Column', default=False) ## @todo fill-codes
#	parser.add_argument('--manual-items', dest='manual', type=bool, help='Update items we manually intervene with. 5/24/2020 COC', default=False)
#	parser.add_argument('--mode', dest='mode', type=str, default='default', help='Mode (default, distant-body so far)')
#	parser.add_argument('--specific-name', dest='specific_name', type=str, help='Specific object name to try') # 6/21/2021 COC

    args = parser.parse_args()
    print(f'UserWeights(input_file={args.input_file})')
    UserWeights(input_file=args.input_file)
