import utils


if __name__ == "__main__":
	ci = utils.CensusIncome("./census_data/")
	(x_tr, y_tr), (x_te, y_te) = ci.load_data()

	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(max_depth=10, random_state=0)
	clf.fit(x_tr, y_tr)
	print(clf.score(x_tr, y_tr))
	print(clf.score(x_te, y_te))

	# constants = utils.Celeb()
	# ds = constants.get_dataset()
	# train_loader, val_loader = ds.make_loaders(batch_size=64, workers=8, shuffle_val=False, only_val=True)