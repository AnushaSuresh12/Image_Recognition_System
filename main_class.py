import matplotlib.pyplot as plt
import error_calculation


# this is the start of the class
if __name__ == "__main__":
    principal_component_list=[]
    average_list=[]
    # Calculate PCA for upto 6 PCA
    principal_list=[]
    for i in range(1,7):
        principal_component_list.append(i)
        accurate_list=error_calculation.calculate_error(i)
        temp=sum(accurate_list)
        avg=float(temp/50)
        average_list.append(avg)
    plt.scatter(principal_component_list,average_list)
    plt.plot(principal_component_list,average_list)
    plt.xlabel("Principal Components")
    plt.ylabel("% Error Rate")
    plt.title("Error Rate vs No.of principal components")
    plt.show()