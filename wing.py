import numpy as np
import json
import matplotlib.pyplot as plt
import copy

from component import Component, Prismoid, PseudoPrismoid, SymmetricAirfoil, DiamondAirfoil

class Wing:
    """A class calculating and containing the mass properties of an aircraft.

    Parameters
    ----------
    input_vars : string or dict , optional
        Must be a .json file or python dictionary.

    Raises
    ------
    TypeError
        If the input_vars type is not a dictionary or the file path to 
        a .json file
    """
    def __init__(self,input_dict={}):

        # retrieve info
        self._retrieve_info(input_dict)

        # organize input info
        self._organize_inputs()

        # initialize components
        self._initialize_components()

        # calculate volume
        self._get_volume()
        
        # if given total mass, determine density
        if not self._given_density_not_mass:
            self._get_density_for_components()


    def _retrieve_info(self,input_dict):
        """A function which retrieves the information and stores it globally.
        """
        
        # store variables from file input dictionary
        self._wing_type = input_dict.get("type","symmetric_airfoil")

        self._side = input_dict.get("side","both")

        self.ID = input_dict.get("ID")

        # store mass / density variables
        if "density" in input_dict:
            self.density = input_dict.get("density")
            self._given_density_not_mass = True
        elif "mass" in input_dict:
            self.mass = input_dict.get("mass")
            self._given_density_not_mass = False
        else: # raise error if neither given
            raise ValueError("No mass / density property given")
        
        # import connect_to info
        connect = input_dict.get("connect_to",{})
        self._connect_ID = connect.get("ID",0)
        self._location = connect.get("location", "tip")
        self._connect_dx = connect.get("dx", 0.0)
        self._connect_dy = connect.get("dy", 0.0)
        self._connect_dz = connect.get("dz", 0.0)
        self._connect_y_offset = connect.get("y_offset", 0.0)

        # import geometry info
        self._b = input_dict.get("semispan",1.0)
        self._dihedral = input_dict.get("dihedral",0.0)
        self._sweep = input_dict.get("sweep",0.0)
        self._chord = input_dict.get("chord",1.0)
        self._thickness = input_dict.get("thickness",0.10)
        self._camber = input_dict.get("camber",0.02)
        self._airfoil = input_dict.get("airfoil","NACA_0008")

        # thickness coefficients
        self._avals = input_dict.get("thickness_distribution_coefficients",\
            "open_trailing_edge")

        # check for incorrect format of values
        if isinstance(self._dihedral,(float,int)):
            val = float(self._dihedral)
            self._dihedral = np.array([[0.0,val],[1.0,val]])
        elif isinstance(self._dihedral,list):
            self._dihedral = np.array(self._dihedral)
        else:
            raise ValueError("dihedral input not correctly formatted")

        if isinstance(self._sweep,(float,int)):
            val = float(self._sweep)
            self._sweep = np.array([[0.0,val],[1.0,val]])
        elif isinstance(self._sweep,list):
            self._sweep = np.array(self._sweep)
        else:
            raise ValueError("sweep input not correctly formatted")

        if isinstance(self._chord,(float,int)):
            val = float(self._chord)
            self._chord = np.array([[0.0,val],[1.0,val]])
        elif isinstance(self._chord,list):
            self._chord = np.array(self._chord)
        else:
            raise ValueError("chord input not correctly formatted")

        if isinstance(self._thickness,(float,int)):
            val = float(self._thickness)
            self._thickness = np.array([[0.0,val],[1.0,val]])
        elif isinstance(self._thickness,list):
            self._thickness = np.array(self._thickness)
        else:
            raise ValueError("thickness input not correctly formatted")

        if isinstance(self._camber,(float,int)):
            val = float(self._camber)
            self._camber = np.array([[0.0,val],[1.0,val]])
        elif isinstance(self._camber,list):
            self._camber = np.array(self._camber)
        else:
            raise ValueError("camber input not correctly formatted")

        if isinstance(self._airfoil,(str)):
            val = self._airfoil
            self._airfoil = [[0.0,val],[1.0,val]]
        elif not isinstance(self._airfoil,list):
            raise ValueError("airfoil input not correctly formatted")


    def _straighten_c_4(self,geo_dict):

        ### "straighten"
        # determine initial and difference angles for calculation
        s_0 = np.deg2rad(geo_dict["sweep"][0])
        s_1 = np.deg2rad(geo_dict["sweep"][1])
        d_0 = np.deg2rad(geo_dict["dihedral"][0])
        d_1 = np.deg2rad(geo_dict["dihedral"][1])

        # calculate shift distance
        if d_1-d_0 != 0.0:
            z_shift = -geo_dict["span"] / (d_1-d_0) * \
                ( np.cos(d_0) - np.cos(d_1) )
            y_shift = geo_dict["span"] / (d_1-d_0) * \
                ( np.sin(d_1) - np.sin(d_0) )
        else:
            y_shift = geo_dict["span"] * np.cos(d_0)
            z_shift = -geo_dict["span"] * np.sin(d_0)
        if s_1-s_0 != 0.0:
            x_shift = geo_dict["span"] / (s_1-s_0) * \
                (np.log(np.abs(np.cos(s_1))) - np.log(np.abs(np.cos(s_0))))
        else:
            x_shift = -geo_dict["span"] * np.tan(s_0)

        
        # calculate corresponding constant angles
        geo_dict["span"] = np.sqrt(z_shift**2. + y_shift**2.)
        geo_dict["dihedral"] = np.rad2deg(np.arctan2(-z_shift, y_shift))
        geo_dict["sweep"] = np.rad2deg(np.arctan2(-x_shift,geo_dict["span"]))

        # determine shift from root quarter chord to tip quarter chord
        if self._side == "left": y_shift = y_shift * -1.
        shift = np.array([x_shift,y_shift,z_shift])
        
        return geo_dict,shift


    def _organize_inputs(self):

        # determine span fraction values
        af_span = [af[0] for af in self._airfoil]
        span_fracs = np.unique(np.concatenate((self._dihedral[:,0], \
            self._sweep[:,0],self._chord[:,0],self._thickness[:,0], \
            self._camber[:,0],af_span)))
        
        # initialize a components dictionary
        self._component_inputs = {}

        # initialize counters
        i_d = i_s = i_c = i_t = i_m = i_a = 0

        # root location
        self._num_spans = span_fracs.shape[0]
        root_location = np.zeros((3,))
        root_location[1] = self._connect_y_offset * 1.0

        # create dictionary for input to component
        for i in range(span_fracs.shape[0]-1):

            # determine span value
            b = (span_fracs[i+1] - span_fracs[i]) * self._b

            # determine root and tip values
            dihedral = self._dihedral[i_d:i_d+2]
            dr = np.interp(span_fracs[i  ],dihedral[:,0],dihedral[:,1])
            dt = np.interp(span_fracs[i+1],dihedral[:,0],dihedral[:,1])

            sweep = self._sweep[i_s:i_s+2]
            sr = np.interp(span_fracs[i  ],sweep[:,0],sweep[:,1])
            st = np.interp(span_fracs[i+1],sweep[:,0],sweep[:,1])

            chord = self._chord[i_c:i_c+2]
            cr = np.interp(span_fracs[i  ],chord[:,0],chord[:,1])
            ct = np.interp(span_fracs[i+1],chord[:,0],chord[:,1])
            
            thickness = self._thickness[i_t:i_t+2]
            tr = np.interp(span_fracs[i  ],thickness[:,0],thickness[:,1])
            tt = np.interp(span_fracs[i+1],thickness[:,0],thickness[:,1])

            camber = self._camber[i_m:i_m+2]
            mr = np.interp(span_fracs[i  ],camber[:,0],camber[:,1])
            mt = np.interp(span_fracs[i+1],camber[:,0],camber[:,1])

            airfoil = self._airfoil[i_a:i_a+2]
            af0 = [af[0] for af in airfoil]
            af1 = [af[1] for af in airfoil]
            if span_fracs[i] >= af0[0] and span_fracs[i] < af0[1]: ar = af1[0]
            else:ar = af1[1]
            if span_fracs[i+1] <= af0[1] and span_fracs[i+1] > af0[0]: at = af1[1]
            else:at = af1[0]

            # create component dictionary
            geo_dict = {}
            geo_dict["span"] = b
            geo_dict["chord"] = [cr,ct]
            geo_dict["thickness"] = [tr,tt]
            geo_dict["camber"] = [mr,mt]
            geo_dict["airfoil"] = [ar,at]
            geo_dict["sweep"] = [sr,st]
            geo_dict["dihedral"] = [dr,dt]

            # "straighten" quarter chord line for each component section
            geo_dict,shift = self._straighten_c_4(geo_dict)

            # add to components input dictionary
            self._component_inputs[i] = {}
            if self._given_density_not_mass:
                self._component_inputs[i]["density"] = self.density
            else:
                self._component_inputs[i]["density"] = 1.0
            self._component_inputs[i]["geometry"] = geo_dict
            self._component_inputs[i]["sweep"] = geo_dict["sweep"]
            self._component_inputs[i]["dihedral"] = geo_dict["dihedral"]
            self._component_inputs[i]["root_location"] = root_location.tolist()
            self._component_inputs[i]["thickness_distribution_coefficients"] =\
                self._avals
            if self._side == "both":
                self._component_inputs[i]["side"] = "right"
            else:
                self._component_inputs[i]["side"] = self._side

            # add to root location
            root_location = root_location + shift
            self._component_inputs[i]["tip_location"] = root_location.tolist()
            
            # update indices
            if i < span_fracs.shape[0]-2:
                i1b = span_fracs[i+1]

                if self._dihedral[i_d+1,0] <= i1b: i_d += 1
                if self._dihedral[i_d+1,0] == self._dihedral[i_d,0]:  i_d += 1

                if self._sweep[i_s+1,0] <= i1b: i_s += 1
                if self._sweep[i_s+1,0] == self._sweep[i_d,0]:  i_s += 1

                if self._chord[i_c+1,0] <= i1b: i_c += 1
                if self._chord[i_c+1,0] == self._chord[i_c,0]:  i_c += 1

                if self._thickness[i_t+1,0] <= i1b: i_t += 1
                if self._thickness[i_t+1,0] == self._thickness[i_t,0]:  i_t +=1

                if self._camber[i_m+1,0] <= i1b: i_m += 1
                if self._camber[i_m+1,0] == self._camber[i_m,0]:  i_m += 1

                if self._airfoil[i_a+1][0] <= i1b: i_a += 1
                if self._airfoil[i_a+1][0] == self._airfoil[i_a][0]:  i_a += 1
        
        # if both wings
        if self._side == "both":
            for i in range(span_fracs.shape[0]-1):
                j = i + span_fracs.shape[0]-1

                self._component_inputs[j] = copy.deepcopy(\
                    self._component_inputs[i])
                self._component_inputs[j]["side"] = "left"
                r_arr = np.array(self._component_inputs[j]["root_location"])
                r_arr[1] *= -1.0
                self._component_inputs[j]["root_location"] = r_arr.tolist()
                t_arr = np.array(self._component_inputs[j]["tip_location"])
                t_arr[1] *= -1.0
                self._component_inputs[j]["tip_location"] = t_arr.tolist()
        
        # add in shift from input
        in_shift = np.copy(np.array([self._connect_dx,self._connect_dy,\
            self._connect_dz]))
        for i in range(len(self._component_inputs)):
            r_arr = np.copy(np.array(self._component_inputs[i]["root_location"]))
            r_arr += in_shift * 1.0
            self._component_inputs[i]["root_location"] = r_arr.tolist()
            t_arr = np.copy(np.array(self._component_inputs[i]["tip_location"]))
            t_arr += in_shift * 1.0
            self._component_inputs[i]["tip_location"] = t_arr.tolist()
        
        # save wing root and tip locations
        self.locations = {}
        self.locations["root"] = np.copy(np.array(\
            self._component_inputs[0]["root_location"]))[:,np.newaxis]
        h = span_fracs.shape[0]-2
        self.locations["tip"] = np.copy(np.array(\
            self._component_inputs[h]["tip_location"]))[:,np.newaxis]


    def _initialize_components(self):

        self._components = {}
        if self._wing_type == "diamond_airfoil":
            for i in self._component_inputs:
                self._components[i] = DiamondAirfoil(self._component_inputs[i])
        elif self._wing_type == "symmetric_airfoil":
            for i in self._component_inputs:
                self._components[i] = SymmetricAirfoil(self._component_inputs[i])
        elif self._wing_type == "pseudo_prismoid":
            for i in self._component_inputs:
                self._components[i] = PseudoPrismoid(self._component_inputs[i])
        elif self._wing_type == "prismoid":
            for i in self._component_inputs:
                self._components[i] = Prismoid(self._component_inputs[i])


    def _get_volume(self):
        
        # sum total volume
        self.volume = 0.0
        for i in self._components:
            self.volume += self._components[i].volume


    def _get_density_for_components(self):

        # save density if given mass
        self.density = self.mass / self.volume
        
        self.update_densities()


    def update_densities(self):
        
        # update individual densities
        for i in self._components:
            self._components[i].density = self.density * 1.


    def get_mass_properties(self):
        """Method which runs the mass properties of each component of the wing
        and returns the total wing mass, cg location, and inertia tensor about
        the cg."""

        # run mass props for each component
        for i in self._components:
            self._components[i].get_mass_properties()
        
        # determine total mass
        self.mass = 0.0
        for i in self._components:
            self.mass += self._components[i].mass
        
        # determine total cg location
        self.cg_location = np.zeros((3,1))
        for i in self._components:
            # component_cg = self._components[i].cg_location + self._components[i].locations["root"]
            self.cg_location += self._components[i].mass * \
                self._components[i].get_cg_location()
            # ##################
            # print("Mass = {:> 10.8f} slugs".format(self._components[i].mass))
            # # print(self._components[i]._root_location)
            # print("Center of mass: (feet)")
            # print("\tX = {:> 10.8f}".format(self._components[i].cg_location[0,0]))
            # print("\tY = {:> 10.8f}".format(self._components[i].cg_location[1,0]))
            # print("\tZ = {:> 10.8f}".format(self._components[i].cg_location[2,0]))
            # I = self._components[i].inertia_tensor * 1.0
            # I[[0,0,1,1,2,2],[1,2,0,2,0,1]] *= -1.0
            # [[Ixx,Ixy,Ixz],[Iyx,Iyy,Iyz],[Izx,Izy,Izz]] = I
            # print("Moment of inertia:( slugs * square feet)")
            # print("\tLxx = {:> 10.8f}\tLxy = {:> 10.8f}\tLxz = {:> 10.8f}".format(Ixx,Ixy,Ixz))
            # print("\tLyx = {:> 10.8f}\tLyy = {:> 10.8f}\tLyz = {:> 10.8f}".format(Iyx,Iyy,Iyz))
            # print("\tLzx = {:> 10.8f}\tLzy = {:> 10.8f}\tLzz = {:> 10.8f}".format(Izx,Izy,Izz))
            # print()
            # ##################
        self.cg_location /= self.mass
        
        # determine total angular momentum
        self.angular_momentum = 0.0
        for i in self._components:
            self.angular_momentum += self._components[i].angular_momentum

        # print(self.cg_location)
        
        # determine total inertia
        self.inertia_tensor = np.zeros((3,3))
        location = self.cg_location
        # location = [[0.0],[0.0],[0.0]]
        for i in self._components:
            self.inertia_tensor += \
                self._components[i].shift_properties_to_location(\
                location)["inertia_tensor"]
        
        # # print(self.inertia_tensor)
        # I = self.inertia_tensor * 1.0
        # I[[0,0,1,1,2,2],[1,2,0,2,0,1]] *= -1.0
        # [[Ixx,Ixy,Ixz],[Iyx,Iyy,Iyz],[Izx,Izy,Izz]] = I
        # print("Moment of inertia:( slugs * square feet)")
        # print("\tIxx = {:> 10.8f}\tIxy = {:> 10.8f}\tIxz = {:> 10.8f}".format(Ixx,Ixy,Ixz))
        # print("\tIyx = {:> 10.8f}\tIyy = {:> 10.8f}\tIyz = {:> 10.8f}".format(Iyx,Iyy,Iyz))
        # print("\tIzx = {:> 10.8f}\tIzy = {:> 10.8f}\tIzz = {:> 10.8f}".format(Izx,Izy,Izz))
        # print()

        # return dictionary of values
        self.properties_dict = {
            "mass" : self.mass,
            "cg_location" : self.cg_location,
            "angular_momentum" : self.angular_momentum,
            "inertia_tensor" : self.inertia_tensor
        }
        return self.properties_dict


    def shift_properties_to_location(self,input_location):
        """Method which determines the mass properties of the given component
        about a given location.
        
        Parameters
        ----------
        input_location : array
            The location about which to determine the mass properties. Must be
            formatted as input_location = [[x],[y],[z]].
        """

        # if input_location is a list, turn into numpy array
        if isinstance(input_location, list):
            input_location = np.array(input_location)
        
        # determine new location
        s = input_location - self.cg_location
        # print(self.cg_location)

        # calculate mass shift from parallel axis theorem
        inner_product = np.matmul(s.T,s)[0,0]
        outer_product = np.matmul(s,s.T)
        I_shift = self.mass*( inner_product * np.eye(3) - outer_product )

        # calculate inertia tensor about new location
        Inew = self.inertia_tensor + I_shift

        # return dictionary of values
        output_dict = {
            "mass" : self.mass,
            "cg_location" : self.cg_location,
            "inertia_tensor" : Inew
        }
        return output_dict
