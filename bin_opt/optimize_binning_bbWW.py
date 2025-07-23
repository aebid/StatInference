import numpy as np
import os
import uproot
import json
import sys
import matplotlib.pyplot as plt
from FLAF.RunKit.envToJson import get_cmsenv

#I had to do this because now the ps_call for MergeResonantLimits wouldn't work in cmsEnv environment
cmssw_env = get_cmsenv(cmssw_path=os.getenv("FLAF_CMSSW_BASE"))
for var in [ 'HOME', 'ANALYSIS_PATH', 'ANALYSIS_DATA_PATH', 'X509_USER_PROXY', 'CENTRAL_STORAGE',
                'ANALYSIS_BIG_DATA_PATH', 'FLAF_CMSSW_BASE', 'FLAF_CMSSW_ARCH' ]:
    if var in os.environ:
        cmssw_env[var] = os.environ[var]


def integrate_uproot(hist, start, end):
    int_values = np.sum(hist.values()[start:end+1])
    int_errors = np.sqrt(np.sum((hist.errors()[start:end+1])**(2.0)))
    return [int_values, int_errors]

def convert_to_json(bins_dict, params):
    json_dict = []
    for key in bins_dict.keys():
        ch = key.split("_")[0]
        cat = "_".join(key.split("_")[1:])
        exists = False
        if not exists:
            entry = {
                    'bins': bins_dict[key],
                    'channels': [ch],
                    'categories': [cat],
                }
            for key, value in params.items():
                entry[key] = [ value ]
            json_dict.append(entry)
                
            
    print(json_dict)
    return json_dict



#This will need to loop over files too
def optimize_binning(shapes_dir, filename, sig_string, params, mass_list, outdir, config, nBinsMax, background_names, important_backgrounds, categories, channels, tag):
    os.makedirs(outdir, exist_ok=True)
    file = uproot.open(os.path.join(shapes_dir, filename))

    limit_history = {}
    limit_list = []

    best_limit = np.inf
    best_bins = None

    nBinsMax = nBinsMax

    #bkg_names = ['TT', 'DY', 'ST', 'VV'] #Moved to an argument
    bkg_names = background_names


    priority_list = []
    bins_dict = {}
    for cat in categories:
        for ch in channels:
            priority_list.append((ch, cat))
            dict_key = f'{ch}_{cat}'
            bins_dict[dict_key] = [0.0, 1.0]
            limit_history[dict_key] = {'nBins': [], 'limits': []}

    for ch, cat in priority_list:
        print("New category, reset best_limit")
        best_limit = np.inf
        ch_cat = f'{ch}/{cat}/'
        dict_key = f'{ch}_{cat}'
        hist_names = [ key.split(';')[0][len(ch_cat):] for key in file.keys() if key.startswith(ch_cat) ]
        hists = { hist_name : file[ch_cat+hist_name] for hist_name in hist_names }
        bkg_names_with_unc = []
        for bkg_name in bkg_names:
            if ch == 'eMu' and bkg_name.startswith('DY'): continue # Pull this from config eventually, for now only bbWW (devin) uses this so its no big deal
            for hist_name in hist_names:
                if hist_name == bkg_name or hist_name.startswith(bkg_name+"_"): # Have to be careful with the names, eg 'TT' and 'TTVV' both startwith 'TT', but we want 'TT_JerUp'
                    bkg_names_with_unc.append(hist_name)

        print(f"Starting ch cat {ch_cat} with mass {mass_list}")
        
        signal_name = sig_string #Maybe update to get from yaml later

        tmp_signal = np.sum(hists[signal_name].values())

        #Start the loop over nBins from 1 to nBinsMax
        patience = 0
        # for nbins in range(1, nBinsMax):
        for nbins in range(5, nBinsMax+1, 5):
            bin_content_goal = tmp_signal/nbins

            print(f"Channel {ch_cat} has a total of {tmp_signal} entries, so with {nbins} bins our per-bin goal is {bin_content_goal}")

            custom_binning = [1.0]
            custom_totals = []
            custom_background_totals = []
            done = False
            nBins = len(hists[signal_name].values())-1
            right_edge = nBins
            big_bin_counter = 1
            minimum_bin = 0
            # If you want to put a cut, change this minimum bin
            minimum_bin = int(0.0*(nBins+1))
            print(f"Using minimum bin {minimum_bin} from nBins {nBins}")
            while not done:
                for left_edge in range(right_edge, minimum_bin-1, -1):
                    tot_integral_signal = integrate_uproot(hists[signal_name], left_edge, nBins)[0]
                    curr_integral_signal = integrate_uproot(hists[signal_name], left_edge, right_edge)[0]
                    integral_bkgs = np.zeros(len(important_backgrounds))
                    integral_bkgs_unc = np.zeros(len(important_backgrounds))
                    tot_bkgs = 0
                    tot_bkgs_unc = 0
                    negative_bins = False
                    for i, bkg_name in enumerate(bkg_names_with_unc):
                        int_bkg = integrate_uproot(hists[bkg_name], left_edge, right_edge)
                        if int_bkg[0] <= 0.0:
                            # print(f"Background {bkg_name} has a negative or empty bin, try again")
                            negative_bins = True
                        if bkg_name in important_backgrounds: #Only check important backgrounds for the individual tests
                            integral_bkgs[i] = int_bkg[0]
                            integral_bkgs_unc[i] = (int_bkg[1])

                        tot_bkgs += int_bkg[0] #But check all backgrounds for the total tests
                        tot_bkgs_unc += (int_bkg[1]**(2.0))

                    if negative_bins:
                        if left_edge == minimum_bin:
                            print("Negative bins all the way to the edge, cannot use this binning")
                            done = True
                            break
                        continue

                    tot_bkgs_unc = np.sqrt(tot_bkgs_unc)


                    if left_edge == minimum_bin:
                        custom_binning.append(left_edge/(nBins+1))
                        custom_totals.append(curr_integral_signal)
                        custom_background_totals.append(tot_bkgs)
                        done = True
                        print("Hit the left edge")
                        break

                    if (tot_integral_signal >= big_bin_counter*bin_content_goal):
                        # print(f"Check out the background unc/tot {integral_bkgs_unc/integral_bkgs}")
                        # print(f"Check the backgrounds for empty bins")
                        for bkg_name in bkg_names_with_unc:
                            this_bkg_int = integrate_uproot(hists[bkg_name], left_edge, right_edge)
                            # print(f"{bkg_name} has {this_bkg_int[0]} entries and {this_bkg_int[1]} uncertainty")

                        # print(f"And the total bkg / tot bkg unc is {tot_bkgs} / {tot_bkgs_unc} = {tot_bkgs/tot_bkgs_unc}")


                    if (tot_integral_signal >= big_bin_counter*bin_content_goal) and (np.max(integral_bkgs_unc/integral_bkgs) <= 0.2) and (tot_bkgs_unc/tot_bkgs <= 0.2):       
                        custom_binning.append(round(hists[signal_name].axis().edges()[left_edge], 3)) # Why am i even rounding here?
                        custom_totals.append(curr_integral_signal)
                        custom_background_totals.append(tot_bkgs)
                        print("At bin ", left_edge, " integral is ", custom_totals[-1])
                        right_edge = left_edge-1
                        big_bin_counter+=1
                        break


            if len(custom_binning) < 2:
                raise ValueError(f"Custom binning for {dict_key} is too small, only {len(custom_binning)} bins found, something went wrong! {custom_binning}")
            print("Found the bin edge set, it is ", custom_binning)
            print("With totals per bin as ", custom_totals)
            print(f"And backgrounds per bin {custom_background_totals}")

            if custom_totals[-1] < 0.2*bin_content_goal:
                print("Final bin was very very small, auto merged!")
                del custom_binning[-2]
                print("New binning is ", custom_binning)

            custom_binning.sort()
            bins_dict[dict_key] = custom_binning

            print("Final dict binnings are ", bins_dict)



            json_dict = convert_to_json(bins_dict, params)


            with open(os.path.join(outdir, f'tmp_binning_{sig_string}.json'), 'w') as f:
                json.dump(json_dict, f)


            #Create the datacards
            datacards_dir = os.path.join(outdir, f'tmp_datacards_{sig_string}')
            binning_file = os.path.join(outdir, f'tmp_binning_{sig_string}.json')


            # Hold up, why am I optimizing by running all channels/categories at once?
            # I should be running each channel/category separately, so that I can optimize the binning for each individually")
            # If we catch std out, then i can avoid printing the create_datacards output!
            quiet_call = ps_call(f"python3 ../dc_make/create_datacards.py --input {shapes_dir} --output {datacards_dir} --config {config} --hist-bins ./{binning_file} --param_values {mass_list}", shell=True, env=cmssw_env, catch_stdout=True)
            # If we want to do this per channel/category, we need to reset datacards_dir to the subfolder
            datacards_dir_category = os.path.join(datacards_dir, f'Run3_2022/{cat}')
            #Find out where the limits will be saved
            # output = ps_call(f"law run MergeResonantLimits --version {tag} --datacards '{datacards_dir}/*.txt' --print-output 0,False", shell=True, catch_stdout=True, split="\n")[1]
            output = ps_call(f"law run MergeResonantLimits --version {tag} --datacards '{datacards_dir_category}/*.txt' --print-output 0,False", shell=True, catch_stdout=True, split="\n")[1]
            limit_file_name = output[-2]
            #Run the limit calculation
            # ps_call(f"law run MergeResonantLimits --version {tag} --datacards '{datacards_dir}/*.txt' --remove-output 3,a,y", shell=True)
            ps_call(f"law run MergeResonantLimits --version {tag} --datacards '{datacards_dir_category}/*.txt' --remove-output 3,a,y", shell=True)

            #Now load the output numpy array and get value [1] (mass, val, up1, down1, up2, down2)
            print(np.load(limit_file_name).files)
            data = np.load(limit_file_name)
            exp_limit = data[data.files[0]][0][1]

            limit_history[dict_key]['nBins'].append(nbins)
            limit_history[dict_key]['limits'].append(exp_limit)
            limit_list.append(exp_limit)

            #If adding a bin does not improve (limit stays the same) then just move on
            if (exp_limit < best_limit) or (nbins == 1):
                if np.isnan(exp_limit): exp_limit = np.inf
                best_limit = exp_limit
                best_bins = bins_dict.copy()
                patience = 0

            # elif (exp_limit == best_limit):
            else:
                print(f"Limit did not change at nbins {nbins}, best limit is still {best_limit}, and current is {exp_limit}")
                # I don't want to let it go to nMax each time, but also don't want to give up too early
                # Introduce patience, allow 3 consecutive 'worse' limits before giving up
                patience += 1
                print(f"Patience now set to {patience}")
                #Need to replace the bin dict because it now has the 'worse' new binning
                bins_dict = best_bins.copy()
                if patience >= 3:
                    #Need to replace the bin dict because it now has the 'worse' new binning
                    # bins_dict = best_bins.copy()
                    break
            # else:
            #     print(f"Did not improve limits at nbins {nbins}, best limit was {best_limit}, and current was {exp_limit}")
            #     break





    print("Finished the limit scan, lets look at the history")
    print(limit_history)

    print(limit_list)
    plt.plot(limit_list)
    plt.yscale('log')
    plot_filename = os.path.join(outdir, f'binopt_history_{sig_string}.pdf')
    plt.savefig(plot_filename)
    plt.close()




if __name__ == '__main__':
        file_dir = os.path.dirname(os.path.abspath(__file__))
        pkg_dir = os.path.dirname(file_dir)
        base_dir = os.path.dirname(pkg_dir)
        pkg_dir_name = os.path.split(pkg_dir)[1]
        if base_dir not in sys.path:
            sys.path.append(base_dir)
        __package__ = pkg_dir_name

        from FLAF.RunKit.run_tools import ps_call
        import argparse
        parser = argparse.ArgumentParser(description='Optimize Binning.')
        parser.add_argument('--input-shapes', required=True, type=str, help="input directory of shape files")
        parser.add_argument('--output', required=True, type=str, help="output directory for new binning, shapes, and datacards")
        parser.add_argument('--config', required=True, type=str, help="configuration file")
        parser.add_argument('--mass-list', required=True, type=str, help="list of mass values to scan over")
        parser.add_argument('--nBinsMax', required=False, type=int, default=10, help="Maximum number of bins to test")
        parser.add_argument('--sig-name-pattern', required=False, type=str, default="ggRadion_HH_bbWW_M{}", help="Name format of signal in the shape files")
        parser.add_argument('--file-name-pattern', required=False, type=str, default="Run3_2022/shape_m{m}.root", help="Name format of input shape files")
        parser.add_argument('--background-names', required=False, type=str, default="TT", help="List of all background processes in the shapes file")
        parser.add_argument('--important-backgrounds', required=False, type=str, default="TT", help="List of background processes to consider for individual statistics")
        parser.add_argument('--categories', required=False, type=str, default="res2b,boost,res1b", help="List of jet categories")
        parser.add_argument('--channels', required=False, type=str, default="eE,eMu,muMu", help="List of lepton channels")
        parser.add_argument('--tag', required=False, type=str, default="dev", help="Name of combine job")

        args = parser.parse_args()


        mass_list = [ int(x) for x in args.mass_list.split(',') ]
        shapes_dir = args.input_shapes
        outdir = args.output
        config = args.config
        nBinsMax = args.nBinsMax
        signal_name_pattern = args.sig_name_pattern
        file_name_pattern = args.file_name_pattern
        background_names = [ x for x in args.background_names.split(',') ]
        important_backgrounds = [ x for x in args.important_backgrounds.split(',') ]
        categories = [ x for x in args.categories.split(',') ]
        channels = [ x for x in args.channels.split(',') ]
        tag = args.tag

        for mass in mass_list:
            optimize_binning(shapes_dir, file_name_pattern.format(m = mass), signal_name_pattern.format(m = mass), {'MX': mass}, mass, outdir, config, nBinsMax, background_names, important_backgrounds, categories, channels, tag)
