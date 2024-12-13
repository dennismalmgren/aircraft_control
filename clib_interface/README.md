g++ -fPIC -I. -c my_sim_integration_lib.cpp -o my_sim_integration_lib.o
g++ -shared -o libmysimintegration.so my_sim_integration_lib.o
