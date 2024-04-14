# M.Sc. Thesis

Repository for Master's thesis project development.

The aim is to implement SVMs on classical and quantum architecture, 
and then to carry out a multi-level comparison to understand 
the advantages and disadvantages of the different paradigm.

## Note 

1) This project is tested with python 3.11, other versions could not work properly.
2) After installing `requirements.txt` you need to install `dwave-inspectorapp` (closed source package) via: 
`pip install dwave-inspectorapp --extra-index=https://pypi.dwavesys.com/simple`
3) For executing on D-Wave Leap solvers you need to create a `.env` file with `DWAVE_API_TOKEN=your-token`