require 'open3'
require 'json'
require 'open-uri'

@commits_url = "commits_url"

@review_comments_url = "review_comments_url"

@original_commit_id = "original_commit_id"

pulls = []

def load_json_url url
	command = "curl #{@auth_key} #{url}"
	stdin, stdout, stderr = Open3.popen3(command)
	json = stdout.read
	myhash = JSON.parse(json)
	return myhash
end

def show_json page
	json = load_json_url page
	formated = JSON.pretty_generate(json)
	return formated
end

def load_diff url
   header = ' -H "Accept: application/vnd.github.diff"'
   command = "curl #{header} #{@auth_key} #{url}"
   stdin, stdout, stderr = Open3.popen3(command)
   json = stdout.read
   return json
end

def load_pulls page
	pulls_requests = "\"https://api.github.com/repos/#{@user_rep}/pulls?state=all&per_page=100&page=#{page}\""

	return load_json_url pulls_requests
end


def load_all_pulls
	pulls = []
	page = 1
	while true
		pulls_tmp = load_pulls(page)

		count = pulls_tmp.count
		puts "#{pulls.count}/#{count}"
		break if count == 0

		pulls += pulls_tmp

		page += 1
	end
	return pulls
end

def prepare patch 
	patch = patch.gsub("diff git", '')
	patch = patch.gsub("/app/src/main/java/com/", '')
 	patch = patch.gsub("\n", '').gsub( /\s+/, " " )

 	return patch
end


def get_patch pull, skip
	commits_json_raw = load_json_url pull[@commits_url]

	commits_json = []

	commits_json_raw.each do |commit|
		commit_sha = commit["sha"]
		if skip and File.file? "patchs/#{commit_sha}-true" or File.file? "patchs/#{commit_sha}-false" or File.file? "trash/#{commit_sha}"
			next
		else
			commits_json << commit
		end
	end

	if (commits_json.size == 0)
		puts "#{pull["number"]} skipped"
		return
	end

	puts "PR #{pull["number"]}: #{commits_json_raw.count} commits"

	review_comments = load_json_url pull[@review_comments_url]

	commits_commented = []

	review_comments.each do |review|
		if (review.count > 0 and review.key? @original_commit_id)
			commitsha = review[@original_commit_id]
			commits_commented << commitsha
		end
	end

	commits_json.each do |commit|
		commit_sha = commit["sha"]

		diff_origin = load_diff commit["url"]
		diff = prepare diff_origin

		if diff.size > 30000 or diff.size == 0 
			puts "skipping diff #{diff.size}"
			File.write("trash/#{commit_sha}", diff.size)
			next
		end
		value = commits_commented.include? commit_sha

		file =  "patchs/#{commit_sha}-#{value}"
		puts "#{file}"
		File.write(file, diff)
	end

end

if (ARGV.size < 3) 
	puts "Params: useranme password user_rep(git_user/git_repo) pull_request_id(optional)"
else
	useranme = ARGV[0]
	password = ARGV[1]
	@user_rep = ARGV[2]

	@auth_key = "--user \"#{useranme}:#{password}\""

	if ARGV.size == 4
		pull_id = ARGV[3]
		puts "Loading patchs from PR #{pull_id}"
		pull = load_json_url "https://api.github.com/repos/#{@user_rep}/pulls/#{pull_id}" 
		get_patch pull, true
	else
		puts "Loading all patchs from github #{@user_rep}"
		pulls = load_all_pulls

		POOL_SIZE = 5

		jobs = Queue.new

		pulls.each do |pull|
			jobs.push pull
		end


		workers = (POOL_SIZE).times.map do
		  Thread.new do
		    begin      
		      while pull = jobs.pop(true)
		        get_patch pull, true
		      end
		    rescue ThreadError
		    end
		  end
		end
		workers.map(&:join)
		puts "patchs loaded"
	end

end
