/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h"     // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

int main(int argc, char **argv)
{

    // Read the inputfile and fill the param structure
    parameters param;
    // Read the input file name from command line
    readInputFile(&param, argc, argv);
    printParameters(&param);
    saveParameters(&param);

    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp = 0.0;

    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);

    // Allocate Fields
    EMfield field;
    field_allocate(&grd, &field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd, &field_aux);

    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for (int is = 0; is < param.ns; is++)
        interp_dens_species_allocate(&grd, &ids[is], is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd, &idn);

    // Allocate Particles
    particles *part = new particles[param.ns];
    particles *partDevice = new particles[param.ns];

    // allocation
    for (int is = 0; is < param.ns; is++)
    {
        particle_allocate(&param, &part[is], is);
        particle_allocate(&param, &partDevice[is], is);
    }

    int ControlValues = 1;

    //*partDevice = *part;

    // Initialization
    initGEM(&param, &grd, &field, &field_aux, part, ids);

    //Initialize copy
    if(ControlValues == 1)
    {
        for(int i = 0; i < part->nop; i++)
        {
            partDevice->x[i] = part->x[i];
            partDevice->y[i] = part->y[i];
            partDevice->z[i] = part->z[i];
            partDevice->u[i] = part->u[i];
            partDevice->v[i] = part->v[i];
            partDevice->w[i] = part->w[i];
        }
    }

    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for (int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++)
    {

        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;

        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn, ids, &grd, param.ns);

        // implicit mover
        iMover = cpuSecond(); // start timer for mover
        for (int is = 0; is < param.ns; is++)
        {
            mover_PC(&part[is], &field, &grd, &param);
            mover_PC_launch_gpu(&partDevice[is], &field, &grd, &param);
        }
        eMover += (cpuSecond() - iMover); // stop timer for mover

        if(ControlValues == 1)
        {
            int countx = 0;
            int county = 0;
            int countz = 0;
            int countu = 0;
            int countv = 0;
            int countw = 0;
            //Float has a certain precision that has to be taken into account
            //Floats can represent 6-7 decimals so we make tolerans 1e-5
            float tolerance = 1e-5;

            for(int i = 0; i < part->nop; i++)
            {
                if(std::abs(part->x[i] - partDevice->x[i]) < tolerance)
                    countx++;
                else
                {
                    std::cout << "Wrong: " << part->x[i] << " and " << partDevice->x[i] << std::endl;
                }
                if(std::abs(part->y[i] - partDevice->y[i]) < tolerance)
                    county++;

                if(std::abs(part->z[i] - partDevice->z[i]) < tolerance)
                    countz++;

                if(std::abs(part->u[i] - partDevice->u[i]) < tolerance)
                    countu++;

                if(std::abs(part->v[i] - partDevice->v[i]) < tolerance)
                    countv++;

                if(std::abs(part->w[i] - partDevice->w[i]) < tolerance)
                    countw++;
            }
            std::cout<<"Amount correct x: " << countx << "/" << part->nop << std::endl;
            std::cout<<"Amount correct y: " << county << "/" << part->nop << std::endl;
            std::cout<<"Amount correct z: " << countz << "/" << part->nop << std::endl;
            std::cout<<"Amount correct u: " << countu << "/" << part->nop << std::endl;
            std::cout<<"Amount correct v: " << countv << "/" << part->nop << std::endl;
            std::cout<<"Amount correct w: " << countw << "/" << part->nop << std::endl;
        }

        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        // interpolate species
        for (int is = 0; is < param.ns; is++)
            interpP2G(&part[is], &ids[is], &grd);
        // apply BC to interpolated densities
        for (int is = 0; is < param.ns; is++)
            applyBCids(&ids[is], &grd, &param);
        // sum over species
        sumOverSpecies(&idn, ids, &grd, param.ns);
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon, &grd, &param);

        // write E, B, rho to disk
        if (cycle % param.FieldOutputCycle == 0)
        {
            VTK_Write_Vectors(cycle, &grd, &field);
            VTK_Write_Scalars(cycle, &grd, ids, &idn);
        }

        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation

    } // end of one PIC cycle

    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd, &field);
    // interp
    interp_dens_net_deallocate(&grd, &idn);

    // Deallocate interpolated densities and particles
    for (int is = 0; is < param.ns; is++)
    {
        interp_dens_species_deallocate(&grd, &ids[is]);
        particle_deallocate(&part[is]);
    }

    // stop timer
    double iElaps = cpuSecond() - iStart;

    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover / param.ncycles << std::endl;
    // std::cout << "   Mover Time gpu / Cycle   (s) = " << eMover2/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp / param.ncycles << std::endl;
    std::cout << "**************************************" << std::endl;

    // exit
    return 0;
}
