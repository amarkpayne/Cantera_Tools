from rmgpy.tools.canteraModel import Cantera, getRMGSpeciesFromUserSpecies
from rmgpy.chemkin import loadChemkinFile
from rmgpy.species import Species
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
import json
import itertools


def create_species_from_smiles(smiles_dictionary):
    """
    Creates a dictionary with user names as keys and specie objects as values

    =========================== =======================================================================
    Input                       Description
    =========================== =======================================================================
    smiles_dictionary           A dictionary with user names as keys and SMILES strings as values
    ===================================================================================================

    =========================== =======================================================================
    Output                      Description
    =========================== =======================================================================
    user_species_dictionary     A dictionary with user names as keys and species objects as values
    ===================================================================================================
    """

    user_species_dictionary = {}
    for (user_name, smiles_string) in smiles_dictionary.iteritems():
        user_species_dictionary[user_name] = Species(label=user_name).fromSMILES(smiles_string)

    return user_species_dictionary


def extract_mol_fractions_from_data(exp_file):
    """Creates a mol_fraction dictionary for run_cantera_job()

    =========================== =======================================================================
    Input                       Description
    =========================== =======================================================================
    exp_file                    String relative path of .json file with mol_per_mass data
    ===================================================================================================

    =========================== =======================================================================
    Output                      Description
    =========================== =======================================================================
    mol_fractions               A dictionary with user names as keys and mol fractions as values
    ===================================================================================================
    """

    with open(exp_file, 'r') as exp_data_file:
        exp_data = json.load(exp_data_file)

    initial_mol_per_mass = 0

    for species in exp_data:
        if species != 'Time':
            initial_mol_per_mass += exp_data[species][0]

    mol_fractions = {}

    for species in exp_data:
        if species != 'Time':
            mol_fractions[species] = (exp_data[species][0]) / initial_mol_per_mass
            if mol_fractions[species] < 1e-8:
                del mol_fractions[species]

    return mol_fractions


def set_species_mol_fractions(mol_fractions, user_species_dictionary):
    """
        Creates a mol_fraction dictionary for cantera simulation

        =========================== =======================================================================
        Input                       Description
        =========================== =======================================================================
        mol_fractions               A dictionary with user names as keys and mol fractions as values
        user_species_dictionary     A dictionary with user names as keys and species objects as values
        ===================================================================================================

        =========================== =======================================================================
        Output                      Description
        =========================== =======================================================================
        cantera_mol_fractions       A dictionary with species objects as keys and mol fractions as values
        ===================================================================================================
    """

    cantera_mol_fractions = {}
    for (user_name, mol_frac) in mol_fractions.iteritems():
        cantera_mol_fractions[user_species_dictionary[user_name]] = mol_frac

    return cantera_mol_fractions


def run_cantera_job(smiles_dictionary,
                    specie_initial_mol_frac,
                    final_time,
                    temp_initial,
                    initial_p,
                    chemkin_file='',
                    species_dictionary_file='',
                    transport_file=None,
                    reactor_type='IdealGasConstPressureTemperatureReactor',
                    time_units='s',
                    temp_units='K',
                    p_units='atm',
                    species_list=None,
                    reaction_list=None,
                    ):
    """General function for running Cantera jobs from chemkin files with common defaults

    =========================== =======================================================================
    Input (Required)            Description
    =========================== =======================================================================
    smiles_dictionary           A dictionary with user names as keys and SMILES strings as values
    specie_initial_mol_frac     A dictionary with user specie names as keys and mol fractions as values
    final_time                  Termination time for the simulation
    temp_initial                Initial temperature for the simulation
    initial_p                   Initial pressure for the simulation
    =========================== =======================================================================
    Inputs with Defaults        Description
    =========================== =======================================================================
    chemkin_file                String relative path of the chem.inp or chem_annotated.inp file
    species_dictionary_file     String relative path of species_dictionary file
    reactor_type                String with Cantera reactor type
    time_units                  Default is s (min and h are also supported)
    temp_units                  Default is K (C is also supported)
    p_units                     Default is atm (bar and Pa are also supported)
    =========================== =======================================================================
    Optional Inputs             Description
    =========================== =======================================================================
    transport_file              String relative path of trans.dat file
    species_list                Output from loadChemkinFile for faster simulation (otherwise generated)
    reaction_list               Output from loadChemkinFile for faster simulation (otherwise generated)
    ===================================================================================================


    =========================== =======================================================================
    Output                      Description
    =========================== =======================================================================
    all_data                    Cantera Simulation Data Object [time, [temp, pressure, spc1, spc2,..]]
    ===================================================================================================
    """
    logging.info('Running a cantera job using the chemkin file {}'.format(chemkin_file))
    
    logging.debug('loading chemkin and species dictionary file')
    cwd = os.getcwd()
    if chemkin_file == '':
        chemkin_file = os.path.join(cwd, 'chem_annotated.inp')

    if species_dictionary_file == '':
        species_dictionary_file = os.path.join(cwd, 'species_dictionary.txt')

    user_species_dictionary = create_species_from_smiles(smiles_dictionary)
    specie_initial_mol_frac = set_species_mol_fractions(specie_initial_mol_frac, user_species_dictionary)

    if (not species_list) or (not reaction_list):
        (species_list, reaction_list) = loadChemkinFile(chemkin_file, species_dictionary_file)

    name_dictionary = getRMGSpeciesFromUserSpecies(user_species_dictionary.values(), species_list)

    mol_fractions = {}
    for (user_name, chemkin_name) in name_dictionary.iteritems():
        try:
            mol_fractions[chemkin_name] = specie_initial_mol_frac[user_name]
        except KeyError:
            logging.debug('{} initial mol fractions set to 0'.format(user_name))

    if temp_units == 'C':
        temp_initial += 273.0

    temp_initial = ([temp_initial], 'K')
    initial_p = ([initial_p], p_units)

    job = Cantera(speciesList=species_list, reactionList=reaction_list, outputDirectory='')
    job.loadChemkinModel(chemkin_file, transportFile=transport_file)
    job.generateConditions([reactor_type], ([final_time], time_units), [mol_fractions], temp_initial, initial_p)
    
    logging.debug('Starting Cantera Simulation')
    all_data = job.simulate()
    all_data = all_data[0]
    logging.info('Cantera Simulation Complete')

    logging.debug('Setting labels to user defined species labels')

    species_index = {}
    for i in range(len(species_list)):
        species_index[species_list[i]] = i+2

    user_index = {}
    for (user_name, specie) in user_species_dictionary.iteritems():
        try:
            user_index[species_index[name_dictionary[specie]]] = user_name
        except KeyError:
            logging.info('{0} is not in the model for {1}'.format(user_name, chemkin_file))

    for (indices, user_label) in user_index.iteritems():
        try:
            all_data[1][indices].label = user_label
        except KeyError:
            pass
    
    return all_data


def change_cantera_data_units(cantera_data, temp_units=None, time_units=None, pressure_units=None, comp_units=None):

    temp_conversions = {'K:C': -273, 'C:K': 273, 'K:K': 0, 'C:C': 0}

    if temp_units:
        conversion_key = '{0}:{1}'.format(cantera_data[1][0].units, temp_units)
        try:
            cantera_data[1][0].data += temp_conversions[conversion_key]
            cantera_data[1][0].units = temp_units
        except KeyError:
            raise Exception('{} units not currently supported'.format(temp_units))

    time_conversions = {'s:h': float(1.0/3600.0), 's:m': float(1.0/60.0), 's:s': 1.0, 'h:s': 3600.0, 'h:m': 60.0,
                        'h:h': 1.0, 'm:h': float(1.0/60.0), 'm:m': 1.0, 'm:s': 60.0}

    if time_units:
        conversion_key = '{0}:{1}'.format(cantera_data[0].units, time_units)
        try:
            cantera_data[0].data *= time_conversions[conversion_key]
            cantera_data[0].units = time_units
        except KeyError:
            raise Exception('{} units not currently supported'.format(time_units))

    pressure_conversions = {'Pa:bar': 1.0e-5, 'Pa:atm': float(1.0/101325.0), 'Pa:Pa': 1.0, 'atm:Pa': 101325.0,
                            'atm:bar': 1.01325, 'atm:atm': 1, 'bar:Pa': 1.0e5, 'bar:atm': float(1/1.01325),
                            'bar:bar': 1.0}

    if pressure_units:
        conversion_key = '{0}:{1}'.format(cantera_data[1][1].units, pressure_units)
        try:
            cantera_data[1][1].data *= pressure_conversions[conversion_key]
            cantera_data[1][1].units = pressure_units
        except KeyError:
            raise Exception('{} units not currently supported'.format(pressure_units))

    if comp_units:
        if (not cantera_data[1][2].units) and (comp_units == 'moles_per_mass'):
            for i in range(len(cantera_data[1][2].data)):
                mass = 0.0
                for j in range(len(cantera_data[1])-2):
                    mass += cantera_data[1][j+2].data[i]*cantera_data[1][j+2].species.molecule[0].getMolecularWeight()

                for j in range(len(cantera_data[1])-2):
                    cantera_data[1][j+2].data[i] /= mass

            for j in range(len(cantera_data[1])-2):
                cantera_data[1][j+2].units = 'moles_per_mass'

    return cantera_data


def model_percent_error(cantera_data, species_to_compare, exp_file):
    """Calculate the percent of the model from experimental data for the species specified"""

    with open(exp_file, 'r') as exp_data_file:
        exp_data = json.load(exp_data_file)

    times = exp_data['Time']
    times = times[1:]

    time_index = {}

    for t in times:
        time_index[t] = 0
        for (index, cantera_times) in enumerate(cantera_data[0].data):
            if (abs(t - cantera_times)) < (abs(t - cantera_data[0].data[time_index[t]])):
                time_index[t] = index

    percent_errors = {}
    for species in species_to_compare:
        percent_errors[species] = 0
        for column in cantera_data[1]:
            if column.label == species:
                for t in range(len(times)):
                    percent_errors[species] += abs(
                        (exp_data[species][t + 1] - column.data[time_index[times[t]]]) * 100) / (
                                               exp_data[species][t + 1])
                percent_errors[species] /= float(len(times))
        print 'The average percent error for {0} is {1}%'.format(species, percent_errors[species])

    total_percent_error = 0
    for error in percent_errors.values():
        total_percent_error += error

    total_percent_error /= float(len(percent_errors))

    print 'The total average percent error is {0}%'.format(total_percent_error)


def model_percent_error_conv(cantera_data, species_to_compare, exp_file, conv_spec):
    """Calculate the percent of the model from experimental data for the species specified"""

    with open(exp_file, 'r') as exp_data_file:
        exp_data = json.load(exp_data_file)

    exp_conv = []
    for t in range(len(exp_data['Time'])):
        exp_conv += [(exp_data[conv_spec][0] - exp_data[conv_spec][t])/(exp_data[conv_spec][0])]

    time_index = {}

    for t in range(len(exp_conv)):
        time_index[t] = 0
        conversion = exp_conv[t]
        for column in cantera_data[1]:
            if column.label == conv_spec:
                best_conv = (column.data[0]-column.data[time_index[t]])/(column.data[0])
                for i in range(len(column.data)):
                    test_conv = (column.data[0]-column.data[i])/(column.data[0])
                    if (abs(test_conv - conversion)) < (abs(best_conv - conversion)):
                        time_index[t] = i
                        best_conv = test_conv

    percent_errors = {}
    for species in species_to_compare:
        percent_errors[species] = 0
        for column in cantera_data[1]:
            if column.label == species:
                for t in range(1, len(exp_conv)):
                    percent_errors[species] += \
                        abs((exp_data[species][t] - column.data[time_index[t]])/exp_data[species][t])*100.0
                percent_errors[species] /= float(len(exp_conv) - 1.0)
        print 'The average percent error for {0} is {1}%'.format(species, percent_errors[species])

    total_percent_error = 0
    for error in percent_errors.values():
        total_percent_error += error
    if conv_spec in species_to_compare:
        total_percent_error -= percent_errors[conv_spec]
        total_percent_error /= float(len(percent_errors) - 1)
    else:
        total_percent_error /= float(len(percent_errors))

    print 'The total average percent error is {0}%'.format(total_percent_error)


def model_mean_square_error(cantera_data, species_to_compare, exp_file):
    """Calculate the MSE of the model from experimental data for the species specified"""

    with open(exp_file, 'r') as exp_data_file:
        exp_data = json.load(exp_data_file)

    times = exp_data['Time']
    times = times[1:]

    time_index = {}

    for t in times:
        time_index[t] = 0
        for (index, cantera_times) in enumerate(cantera_data[0].data):
            if (abs(t - cantera_times)) < (abs(t - cantera_data[0].data[time_index[t]])):
                time_index[t] = index

    mean_square_errors = {}
    for species in species_to_compare:
        mean_square_errors[species] = 0
        for column in cantera_data[1]:
            if column.label == species:
                for t in range(len(times)):
                    mean_square_errors[species] += (exp_data[species][t + 1] - column.data[time_index[times[t]]]) ** 2.0
                mean_square_errors[species] /= float(len(times))
        print 'The mean square error for {0} is {1} {2}'.format(species, mean_square_errors[species],
                                                                cantera_data[1][2].units)

    total_mse = 0
    for error in mean_square_errors.values():
        total_mse += error

    total_mse /= float(len(mean_square_errors))

    print 'The total average mean square error is {0} {1}'.format(total_mse, cantera_data[1][2].units)


def concentration_profile_plot(cantera_data, species_to_plot, exp_file=None, err_bar_file=None, logscale=False):
    """Creates a publishable quality plot of species concentrations versus time"""

    color_tup = ('b', 'g', 'darkorange', 'r', '#33EECC', 'k', 'm')
    color_cycle = itertools.cycle(color_tup)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14,
            }

    matplotlib.rc('font', **font)  # Use a bold font
    matplotlib.rc('axes', linewidth=3)  # Increase weight of figure border

    times = cantera_data[0].data
    profiles = {}
    colors = {}
    for species in species_to_plot:
        for column in cantera_data[1]:
            if column.label == species:
                profiles[species] = column.data
                colors[species] = color_cycle.next()

    plt.figure(edgecolor='black', figsize=(10, 7))
    plt.xlabel('Time [{}]'.format(cantera_data[0].units), fontsize=18, weight='bold')
    plt.xlim([0, times[-1] * 1.05])
    if not cantera_data[1][2].units:
        plt.ylabel('Mol Fraction [-]', fontsize=18, weight='bold')
    elif cantera_data[1][2].units == 'moles_per_mass':
        plt.ylabel('Moles per Mass [mol/kg]', fontsize=18, weight='bold')

    for (label, profile) in profiles.iteritems():
        plt.plot(times, profile, linewidth=3.0, color=colors[label])

    if exp_file:
        with open(exp_file, 'r') as exp_data_file:
            exp_data = json.load(exp_data_file)

        if err_bar_file:
            with open(err_bar_file, 'r') as err_file:
                err_data = json.load(err_file)

            for species in species_to_plot:
                for i in range(len(exp_data['Time'])):
                    plt.plot(exp_data['Time'][i], exp_data[species][i], 'o', color=colors[species])
                    plt.errorbar(exp_data['Time'][i], exp_data[species][i], err_data[species][i], 0,
                                 color=colors[species], markersize=0, capsize=5)

        else:
            for species in species_to_plot:
                for i in range(len(exp_data['Time'])):
                    plt.plot(exp_data['Time'][i], exp_data[species][i], 'o', color=colors[species])

    plt.legend(profiles.keys(), loc='lower center', bbox_to_anchor=(0.5, 1), ncol=min(len(profiles), 3))

    plt.gca().tick_params(direction='in', width=3, length=10)  # Place ticks on the inside of the plot
    if not logscale:
        current_yticks = plt.yticks()[0]  # Get the current tick values
        plt.gca().set_ylim(top=current_yticks[-1])  # Set the top most tick flush with the top of the figure
        plt.yticks(current_yticks)  # Set the yticks to the same values as before (previous step changes these)
        plt.gca().set_ylim(bottom=0)  # Set the lowest concentration at zero
    else:
        plt.yscale('log')
        finals = []
        for concentrations in profiles.values():
            finals += [np.log10(concentrations[-1])]
        low = float(round(min(finals)) - 1)
        if 10.0**low < 0.0001:
            plt.gca().set_ylim(bottom=0.0001)
        else:
            plt.gca().set_ylim(bottom=10.0**low)

    plt.show()


def get_conversion_from_time(cantera_data, time, conversions):
    t_index = 0
    t_best = cantera_data[0].data[t_index]

    for i in range(len(cantera_data[0].data)):
        if (abs(cantera_data[0].data[i] - time)) < (abs(t_best - time)):
            t_index = i
            t_best = cantera_data[0].data[i]

    return conversions[t_index]


def concentration_conversion_plot(cantera_data, species_to_plot, conversion_spec, exp_file=None, err_bar_file=None,
                                  logscale=False):
    """Creates a publishable quality plot of species concentrations versus time"""

    color_tup = ('b', 'g', 'darkorange', 'r', '#33EECC', 'k', 'm')
    color_cycle = itertools.cycle(color_tup)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14,
            }

    matplotlib.rc('font', **font)  # Use a bold font
    matplotlib.rc('axes', linewidth=3)  # Increase weight of figure border

    profiles = {}
    colors = {}
    for species in species_to_plot:
        for column in cantera_data[1]:
            if column.label == species:
                profiles[species] = column.data
                colors[species] = color_cycle.next()

    conversions = []
    for i in range(len(cantera_data[1]) - 2):
        if cantera_data[1][i + 2].label == conversion_spec:
            x_init = cantera_data[1][i + 2].data[0]
            for j in range(len(cantera_data[1][i + 2].data)):
                conversions += [(x_init - cantera_data[1][i + 2].data[j]) / x_init]

    plt.figure(edgecolor='black', figsize=(10, 7))
    plt.xlabel('{} Conversion [-]'.format(conversion_spec), fontsize=18, weight='bold')
    plt.xlim([0, conversions[-1] * 1.05])
    if not cantera_data[1][2].units:
        plt.ylabel('Mol Fraction [-]', fontsize=18, weight='bold')
    elif cantera_data[1][2].units == 'moles_per_mass':
        plt.ylabel('Moles per Mass [mol/kg]', fontsize=18, weight='bold')

    for (label, profile) in profiles.iteritems():
        plt.plot(conversions, profile, linewidth=3.0, color=colors[label])

    if exp_file:
        with open(exp_file, 'r') as exp_data_file:
            exp_data = json.load(exp_data_file)

        plt.xlim(
            [0, ((exp_data[conversion_spec][0] - exp_data[conversion_spec][-1])/exp_data[conversion_spec][0])*1.05]
        )

        if err_bar_file:
            with open(err_bar_file, 'r') as err_file:
                err_data = json.load(err_file)

            for species in species_to_plot:
                for i in range(len(exp_data['Time'])):
                    conv = (exp_data[conversion_spec][0]-exp_data[conversion_spec][i])/(exp_data[conversion_spec][0])
                    plt.plot(conv, exp_data[species][i], 'o', color=colors[species])
                    plt.errorbar(conv, exp_data[species][i], err_data[species][i], 0,
                                 color=colors[species], markersize=0, capsize=5)

        else:
            for species in species_to_plot:
                for i in range(len(exp_data['Time'])):
                    conv = (exp_data[conversion_spec][0]-exp_data[conversion_spec][i])/(exp_data[conversion_spec][0])
                    plt.plot(conv, exp_data[species][i], 'o', color=colors[species])

    plt.legend(profiles.keys(), loc='lower center', bbox_to_anchor=(0.5, 1), ncol=min(len(profiles), 3))

    plt.gca().tick_params(direction='in', width=3, length=10)  # Place ticks on the inside of the plot
    if not logscale:
        current_yticks = plt.yticks()[0]  # Get the current tick values
        plt.gca().set_ylim(top=current_yticks[-1])  # Set the top most tick flush with the top of the figure
        plt.yticks(current_yticks)  # Set the yticks to the same values as before (previous step changes these)
        plt.gca().set_ylim(bottom=0)  # Set the lowest concentration at zero

    else:
        plt.yscale('log')
        finals = []
        for concentrations in profiles.values():
            finals += [np.log10(concentrations[-1])]
        low = float(round(min(finals)) - 1)
        if 10.0**low < 0.0001:
            plt.gca().set_ylim(bottom=0.0001)
        else:
            plt.gca().set_ylim(bottom=10.0**low)

    plt.show()
