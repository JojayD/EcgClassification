from ECGClassifier import ECGClassifier
from data.ecgdata import mock_data
def main():


	ecg = ECGClassifier("bert-base-uncased")
	d = ecg.load_data(mock_data)
	ecg.pre_process_data(d)

if __name__ == '__main__':
	main()
